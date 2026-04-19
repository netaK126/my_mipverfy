using JuMP
using Memento
using MathOptInterface

"""
$(SIGNATURES)

Checks whether a JuMPLinearType is constant (and thus has no model associated)
with it. This can only be true if it is an affine expression with no stored
variables.
"""
function is_constant(x::JuMP.AffExpr)
    # TODO (vtjeng): Determine whether there is a built-in function for this as of JuMP>=0.19
    all(values(x.terms) .== 0)
end

function is_constant(x::JuMP.VariableRef)
    false
end

function get_tightening_algorithm(
    x::JuMPLinearType,
    nta::Union{TighteningAlgorithm,Nothing},
)::TighteningAlgorithm
    if is_constant(x)
        return interval_arithmetic
    elseif !(nta === nothing)
        return nta
    else
        # x is not constant, and thus x must have an associated model
        model = owner_model(x)
        return !haskey(model.ext, :MIPVerify) ? DEFAULT_TIGHTENING_ALGORITHM :
               model.ext[:MIPVerify].tightening_algorithm
    end
end

@enum BoundType lower_bound_type = -1 upper_bound_type = 1
#! format: off
bound_f = Dict(
    lower_bound_type => lower_bound,
    upper_bound_type => upper_bound
)
bound_obj = Dict(
    lower_bound_type => MathOptInterface.MIN_SENSE,
    upper_bound_type => MathOptInterface.MAX_SENSE
)
bound_delta_f = Dict(
    lower_bound_type => (b, b_0) -> b - b_0,
    upper_bound_type => (b, b_0) -> b_0 - b
)
bound_operator = Dict(
    lower_bound_type => >=,
    upper_bound_type => <=
)
#! format: on

"""
$(SIGNATURES)

Context manager for running `f` on `model`. If `should_relax_integrality` is true, the 
integrality constraints are relaxed before `f` is run and re-imposed after.
"""
function relax_integrality_context(f, model::Model, should_relax_integrality::Bool)
    if should_relax_integrality
        undo_relax = relax_integrality(model)
    end
    r = f(model)
    if should_relax_integrality
        undo_relax()
    end
    return r
end

"""
$(SIGNATURES)

Optimizes the value of `objective` based on `bound_type`, with `b_0`, computed via interval
arithmetic, as a backup.

- If an optimal solution is reached, we return the objective value. We also verify that the 
  objective found is better than the bound `b_0` provided; if this is not the case, we throw an
  error.
- If we reach the user-defined time limit, we compute the best objective bound found. We compare 
  this to `b_0` and return the better result.
- For all other solve statuses, we warn the user and report `b_0`.
"""
function tight_bound_helper(m::Model, bound_type::BoundType, objective::JuMPLinearType, b_0::Number
)
    if reuse_bounds_conf.is_reuse_bounds_and_deps
        b = reuse_bounds_conf.reusable_bounds[reuse_bounds_conf.reusable_indexes]
        reuse_bounds_conf.reusable_indexes += 1
        return b
    end
    @objective(m, bound_obj[bound_type], objective)
    optimize!(m)
    status = JuMP.termination_status(m)
    if status == MathOptInterface.OPTIMAL
        b = JuMP.objective_value(m)
        db = bound_delta_f[bound_type](b, b_0)
        if db < -1e-8
            Memento.warn(MIPVerify.LOGGER, "Δb = $(db)")
            Memento.error(
                MIPVerify.LOGGER,
                "Δb = $(db). Tightening via interval arithmetic should not give a better result than an optimal optimization.",
            )
        end
        append!(reuse_bounds_conf.reusable_bounds, b)
        return b
    elseif status == MathOptInterface.TIME_LIMIT
        append!(reuse_bounds_conf.reusable_bounds, b_0)
        return b_0
    else
        Memento.warn(
            MIPVerify.LOGGER,
            "Unexpected solve status $(status); using interval_arithmetic to obtain bound.",
        )
        append!(reuse_bounds_conf.reusable_bounds, b_0)
        return b_0
    end
end

"""
Calculates a tight bound of type `bound_type` on the variable `x` using the specified
tightening algorithm `nta`.

If an upper bound is proven to be below cutoff, or a lower bound is proven to above cutoff,
the algorithm returns early with whatever value was found.
"""
function tight_bound(
    x::JuMPLinearType,
    nta::Union{TighteningAlgorithm,Nothing},
    bound_type::BoundType,
    cutoff::Real,
)
    tightening_algorithm = get_tightening_algorithm(x, nta)
    b_0 = bound_f[bound_type](x)
    if tightening_algorithm == interval_arithmetic ||
       is_constant(x) ||
       bound_operator[bound_type](b_0, cutoff)
        return b_0
    end
    should_relax_integrality = (tightening_algorithm == lp)
    # x is not constant, and thus x must have an associated model
    bound_value = return relax_integrality_context(owner_model(x), should_relax_integrality) do m
        tight_bound_helper(m, bound_type, x, b_0)
    end

    return bound_value
end

function tight_upperbound(
    x::JuMPLinearType;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff::Real = -Inf,
)
    tight_bound(x, nta, upper_bound_type, cutoff)
end

function tight_lowerbound(
    x::JuMPLinearType;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff::Real = Inf,
)
    tight_bound(x, nta, lower_bound_type, cutoff)
end

function log_gap(m::JuMP.Model)
    gap = abs(1 - JuMP.objective_bound(m) / JuMP.objective_value(m))
    Memento.info(
        MIPVerify.LOGGER,
        "Hit user limit during solve to determine bounds. Multiplicative gap was $gap.",
    )
end

function relu(x::T)::T where {T<:Real}
    return max(zero(T), x)
end

function relu(x::AbstractArray{T}) where {T<:Real}
    return relu.(x)
end

function relu(x::T, l::Real, u::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    global network_version
    global layer_counter
    global nueron_counter
    
    neurons_names.neuron += 1

    if u < l
        # TODO (vtjeng): This check is in place in case of numerical error in the calculation of bounds.
        # See sample number 4872 (1-indexed) when verified on the lp0.4 network.
        Memento.warn(
            MIPVerify.LOGGER,
            "Inconsistent upper and lower bounds: u-l = $(u - l) is negative. Attempting to use interval arithmetic bounds instead ...",
        )
        u = upper_bound(x)
        l = lower_bound(x)
    end

    # Tighten N2(x) bounds using derived N1 + diff bounds
    if bound_n2_relu_using_zonotope && (network_version == "n2_org" || network_version == "n2_pert")
        m_idx = layer_counter
        k_idx = neurons_names.neuron
        if m_idx >= 1 && m_idx <= length(n2_derived_preact_up_bounds) && k_idx >= 1 && k_idx <= length(n2_derived_preact_up_bounds[m_idx])
            u_derived = n2_derived_preact_up_bounds[m_idx][k_idx]
            l_derived = n2_derived_preact_down_bounds[m_idx][k_idx]
            u = min(u, u_derived)
            l = max(l, l_derived)
        end
    end

    # Tighten N2(x') bounds using N1 preact + composed bounds
    if bound_n2_xp_using_composed && network_version == "n2_pert"
        m_idx = layer_counter
        k_idx = neurons_names.neuron
        if !isempty(n2_xp_derived_preact_up_bounds) && m_idx >= 1 && m_idx <= length(n2_xp_derived_preact_up_bounds) &&
           k_idx >= 1 && k_idx <= length(n2_xp_derived_preact_up_bounds[m_idx])
            u_derived = n2_xp_derived_preact_up_bounds[m_idx][k_idx]
            l_derived = n2_xp_derived_preact_down_bounds[m_idx][k_idx]
            u = min(u, u_derived)
            l = max(l, l_derived)
        end
    end

    # ── Advanced-standard: tighten N2 bounds using N1 + diff bounds ─────
    # After solving N1's standard MIP, we have N1's pre-activation bounds
    # in n1_neuron_bounds. Combined with diff bounds from
    # compute_diff_and_comp_bounds(nn1, nn2, ...), we derive tighter N2 bounds:
    #   z_N2_pre ∈ [l_N1 + diff_down, u_N1 + diff_up]
    # Sound by interval arithmetic (see plan for full proof).
    # These tighter bounds may eliminate binary variables when a split neuron
    # becomes provably stable (u_tight <= 0 or l_tight >= 0).
    if !isempty(n1_neuron_bounds) && (network_version == "org" || network_version == "perturbation")
        key = (neurons_names.layer, neurons_names.neuron)
        if haskey(n1_neuron_bounds, key)
            (n1_u, n1_l) = n1_neuron_bounds[key]
            m_idx = layer_counter
            k_idx = neurons_names.neuron
            if !isempty(relu_diff_up_bounds) && m_idx >= 1 && m_idx <= length(relu_diff_up_bounds) &&
               k_idx >= 1 && k_idx <= length(relu_diff_up_bounds[m_idx])
                diff_up_val = relu_diff_up_bounds[m_idx][k_idx]
                diff_down_val = relu_diff_down_bounds[m_idx][k_idx]
                u_tight = n1_u + diff_up_val
                l_tight = n1_l + diff_down_val
                u = min(u, u_tight)
                l = max(l, l_tight)
            end
        end
    end

    # ── Source B: absolute N2 zonotope with per-layer N1 tightening ─────
    # Populated by compute_n2_bounds_zonotope_with_n1_tighten when
    # --adv_std_zono_bounds is active. Empty otherwise, so this block is a
    # no-op in the default path. Sound as an over-approximation of N2's
    # value set; intersection with the existing [l, u] preserves feasibility.
    if !isempty(n2_abs_up_bounds) && (network_version == "org" || network_version == "perturbation")
        m_idx = layer_counter
        k_idx = neurons_names.neuron
        if m_idx >= 1 && m_idx <= length(n2_abs_up_bounds) &&
           k_idx >= 1 && k_idx <= length(n2_abs_up_bounds[m_idx])
            u = min(u, n2_abs_up_bounds[m_idx][k_idx])
            l = max(l, n2_abs_down_bounds[m_idx][k_idx])
        end
    end

    # ── Source C: N1-probe LP bounds (--adv_std_n1_probe=lp) ────────────
    # Populated by compute_n2_bounds_n1_probe_lp via per-neuron OBBT on a
    # joint LP-relaxed (N1 + N2) model. Two separate arrays because the
    # probe runs independently for the clean-input ("org") and perturbed-
    # input ("perturbation") network_version passes. Empty by default;
    # block is a no-op when the flag is off.
    if network_version == "org" && !isempty(n2_probe_up_bounds_org)
        m_idx = layer_counter
        k_idx = neurons_names.neuron
        if m_idx >= 1 && m_idx <= length(n2_probe_up_bounds_org) &&
           k_idx >= 1 && k_idx <= length(n2_probe_up_bounds_org[m_idx])
            u = min(u, n2_probe_up_bounds_org[m_idx][k_idx])
            l = max(l, n2_probe_down_bounds_org[m_idx][k_idx])
        end
    elseif network_version == "perturbation" && !isempty(n2_probe_up_bounds_pert)
        m_idx = layer_counter
        k_idx = neurons_names.neuron
        if m_idx >= 1 && m_idx <= length(n2_probe_up_bounds_pert) &&
           k_idx >= 1 && k_idx <= length(n2_probe_up_bounds_pert[m_idx])
            u = min(u, n2_probe_up_bounds_pert[m_idx][k_idx])
            l = max(l, n2_probe_down_bounds_pert[m_idx][k_idx])
        end
    end

    if u <= 0
        # rectified value is always 0
        return zero(T)
    elseif u == l
        return one(T) * l
    elseif u < l
        error(
            MIPVerify.LOGGER,
            "Inconsistent upper and lower bounds even after using only interval arithmetic: u-l = $(u - l) is negative",
        )
    elseif l >= 0
        # rectified value is always x
        return x
    else
        model = owner_model(x)

        # Helper for triangle relaxation-gap area scoring (used by both relaxation paths)
        _tri_gap(l_val, u_val) = (l_val >= 0.0 || u_val <= 0.0) ? 0.0 : u_val * (-l_val) / (2.0 * (u_val - l_val))

        # ── N2-only perturbation relaxation (--no_n1_binaries_and_relaxtions_only_on_n2) ──
        # Relax N2(x_p) by conditioning on N2(x) binary (a_n2_org) using
        # perturbation bounds through N2 (z_n2_pert - z_n2_org).
        # N2(x) stays exact; N1(x) is LP-relaxed (handled below in standard encoding).
        if no_n1_binaries_and_relaxtions_only_on_n2 && network_version == "n2_pert"
            m_idx = layer_counter
            k_idx = neurons_names.neuron

            if m_idx <= length(relu_n2pert_up_bounds) && k_idx <= length(relu_n2pert_up_bounds[m_idx])
                u_int = relu_n2pert_up_bounds[m_idx][k_idx]
                l_int = relu_n2pert_down_bounds[m_idx][k_idx]
                int_width = u_int - l_int

                relax_score = int_width
                if relaxation_gap_area
                    if m_idx <= length(n2_preact_up_bounds) && k_idx <= length(n2_preact_up_bounds[m_idx])
                        u_pre_tmp = n2_preact_up_bounds[m_idx][k_idx]
                        l_pre_tmp = n2_preact_down_bounds[m_idx][k_idx]
                        lA_tmp = l_int;              uA_tmp = u_pre_tmp + u_int
                        lI_tmp = l_pre_tmp + l_int;  uI_tmp = u_int
                        relax_score = max(_tri_gap(lA_tmp, uA_tmp), _tri_gap(lI_tmp, uI_tmp))
                    end
                end

                if relax_score < relaxation_threshold
                    global relaxation_condition_count += 1
                    # Condition on N2(x) binary instead of N1(x)
                    a_pre_name = string("n2_org", "a_layerCount", layer_counter,
                                        "_neuronCount", nueron_counter,
                                        "_", m_idx, "_", k_idx)
                    a_pre = variable_by_name(model, a_pre_name)

                    if a_pre !== nothing &&
                       m_idx <= length(n2_preact_up_bounds) &&
                       k_idx <= length(n2_preact_up_bounds[m_idx])

                        u_pre = n2_preact_up_bounds[m_idx][k_idx]
                        l_pre = n2_preact_down_bounds[m_idx][k_idx]

                        lA = l_int;          uA = u_pre + u_int   # active   (a_n2_org=1)
                        lI = l_pre + l_int;  uI = u_int           # inactive (a_n2_org=0)

                        av = JuMP.all_variables(model)
                        push!(layers_info_dict,
                              (neurons_names.layer, neurons_names.neuron) => (u, l, length(av)))

                        x_rect = @variable(model)
                        set_lower_bound(x_rect, 0.0)
                        set_upper_bound(x_rect, max(max(0.0, uA), max(0.0, uI)))
                        set_name(x_rect, string(network_version, "x_rect",
                                                "_layerCount", layer_counter,
                                                "_neuronCount", nueron_counter,
                                                "_", m_idx, "_", k_idx))

                        M = u + (-l)

                        @constraint(model, x_rect >= 0)
                        @constraint(model, x_rect >= x)

                        if lA >= 0.0
                            @constraint(model, x_rect <= x + M * (1 - a_pre))
                        elseif uA <= 0.0
                            @constraint(model, x_rect <= M * (1 - a_pre))
                        else
                            @constraint(model, x_rect <=
                                (uA / (uA - lA)) * (x - lA) + M * (1 - a_pre))
                        end

                        if lI >= 0.0
                            @constraint(model, x_rect <= x + M * a_pre)
                        elseif uI <= 0.0
                            @constraint(model, x_rect <= M * a_pre)
                        else
                            @constraint(model, x_rect <=
                                (uI / (uI - lI)) * (x - lI) + M * a_pre)
                        end

                        return x_rect
                    end
                end
            end
        end

        # ── Conditional-triangle relaxations (n2_org and n2_pert passes) ────────
        # BRIDGE paper Section 5, eqs. (4) and (6).
        #
        # Both relaxations condition on a_n1_org (Npre's binary) and use Npre's
        # pre-activation bounds [l_pre, u_pre].  They differ only in which
        # interval bounds are used:
        #
        #   n2_org (activation relaxation, eq. 4):
        #     interval = diff bounds [l_diff, u_diff] = z_n2_org - z_n1_org
        #     threshold: u_diff - l_diff < T_relax
        #
        #   n2_pert (perturbation relaxation, eq. 6):
        #     interval = composed bounds [l_comp, u_comp] = diff + pert
        #                                               = z_n2_pert - z_n1_org
        #     threshold: u_comp - l_comp < T_relax
        #
        # Conditional intervals (same formula for both, with their respective bounds):
        #   Active   (a_n1_org=1): zˆ ∈ [l_int,       u_pre + u_int]
        #   Inactive (a_n1_org=0): zˆ ∈ [l_pre + l_int, u_int      ]
        #
        # l < 0 < u is guaranteed (split case). Big-M = u + |l|.
        # Skip when no_n1_binaries_and_relaxtions_only_on_n2 is active (N1 binaries are LP-relaxed).
        if use_relaxations && !no_n1_binaries_and_relaxtions_only_on_n2 && (network_version == "n2_org" || network_version == "n2_pert" || network_version == "perturbation")
            m_idx = layer_counter         # ReLU layer index within current network (1-based, reset per pass)
            k_idx = neurons_names.neuron  # neuron index within the layer (1-based)
            #NETA
            # Select the correct interval bounds for this pass
            # n2_org (transfer): diff bounds;  n2_pert / perturbation: composed/pert bounds
            bounds_up   = (network_version == "n2_org") ? relu_diff_up_bounds   : relu_comp_up_bounds
            bounds_down = (network_version == "n2_org") ? relu_diff_down_bounds : relu_comp_down_bounds

            if m_idx <= length(bounds_up) && k_idx <= length(bounds_up[m_idx])
                u_int = bounds_up[m_idx][k_idx]
                l_int = bounds_down[m_idx][k_idx]
                int_width = u_int - l_int

                # Decide whether to relax: either by interval width or by gap area
                relax_score = int_width  # default: interval width
                if relaxation_gap_area
                    # Method 2: triangle relaxation-gap area scoring
                    # Requires preact bounds to compute conditional intervals
                    if m_idx <= length(n1_preact_up_bounds) && k_idx <= length(n1_preact_up_bounds[m_idx])
                        u_pre_tmp = n1_preact_up_bounds[m_idx][k_idx]
                        l_pre_tmp = n1_preact_down_bounds[m_idx][k_idx]
                        lA_tmp = l_int;              uA_tmp = u_pre_tmp + u_int
                        lI_tmp = l_pre_tmp + l_int;  uI_tmp = u_int
                        relax_score = max(_tri_gap(lA_tmp, uA_tmp), _tri_gap(lI_tmp, uI_tmp))
                    end
                end

                if relax_score < relaxation_threshold
                    global relaxation_condition_count += 1
                    # Look up the predecessor network's binary for this neuron:
                    #   transfer mode (n2_org/n2_pert): prefix = "n1_org"
                    #   standard mode (perturbation):   prefix = "org"
                    a_pre_prefix = (network_version == "perturbation") ? "org" : "n1_org"
                    a_pre_name = string(a_pre_prefix, "a_layerCount", layer_counter,
                                        "_neuronCount", nueron_counter,
                                        "_", m_idx, "_", k_idx)
                    a_pre = variable_by_name(model, a_pre_name)

                    if a_pre !== nothing &&
                       m_idx <= length(n1_preact_up_bounds) &&
                       k_idx <= length(n1_preact_up_bounds[m_idx])

                        u_pre = n1_preact_up_bounds[m_idx][k_idx]
                        l_pre = n1_preact_down_bounds[m_idx][k_idx]

                        # Conditional intervals (paper eqs. 4 / 6)
                        lA = l_int;          uA = u_pre + u_int   # active   (a_n1_org=1)
                        lI = l_pre + l_int;  uI = u_int           # inactive (a_n1_org=0)

                        av = JuMP.all_variables(model)
                        push!(layers_info_dict,
                              (neurons_names.layer, neurons_names.neuron) => (u, l, length(av)))

                        x_rect = @variable(model)
                        set_lower_bound(x_rect, 0.0)
                        set_upper_bound(x_rect, max(max(0.0, uA), max(0.0, uI)))
                        set_name(x_rect, string(network_version, "x_rect",
                                                "_layerCount", layer_counter,
                                                "_neuronCount", nueron_counter,
                                                "_", m_idx, "_", k_idx))

                        M = u + (-l)  # u + |l|, l < 0 guaranteed

                        # Base constraints (always hold)
                        @constraint(model, x_rect >= 0)
                        @constraint(model, x_rect >= x)

                        # Active-case upper bound (binding when a_n1_org=1, relaxed when 0)
                        if lA >= 0.0
                            @constraint(model, x_rect <= x + M * (1 - a_pre))
                        elseif uA <= 0.0
                            @constraint(model, x_rect <= M * (1 - a_pre))
                        else
                            @constraint(model, x_rect <=
                                (uA / (uA - lA)) * (x - lA) + M * (1 - a_pre))
                        end

                        # Inactive-case upper bound (binding when a_n1_org=0, relaxed when 1)
                        if lI >= 0.0
                            @constraint(model, x_rect <= x + M * a_pre)
                        elseif uI <= 0.0
                            @constraint(model, x_rect <= M * a_pre)
                        else
                            @constraint(model, x_rect <=
                                (uI / (uI - lI)) * (x - lI) + M * a_pre)
                        end

                        return x_rect
                    end
                end
            end
        end

        # ── Transfer-aware: replace N2 binary with triangle relaxation when N1 neuron is stable ──
        # If N1's corresponding neuron has a known activation status (always active or
        # always inactive), N2's activation is tightly constrained by the diff bounds.
        # We can replace N2's binary variable with a triangle LP relaxation — sound
        # (delta_diff >= exact) and tight when diff bounds are narrow.
        # Standard mode cannot do this (no reference network).
        if n1_stability_relax_threshold >= 0 && (network_version == "n2_org" || network_version == "n2_pert")
            m_idx = layer_counter
            k_idx = neurons_names.neuron
            if !isempty(n1_preact_up_bounds) && m_idx >= 1 && m_idx <= length(n1_preact_up_bounds) &&
               k_idx >= 1 && k_idx <= length(n1_preact_up_bounds[m_idx])
                n1_l = n1_preact_down_bounds[m_idx][k_idx]
                n1_u = n1_preact_up_bounds[m_idx][k_idx]
                n1_is_stable = (n1_l >= 0.0) || (n1_u <= 0.0)
                if n1_is_stable
                    tri_gap = _tri_gap(l, u)
                    if tri_gap <= n1_stability_relax_threshold
                        # Triangle LP relaxation (no binary variable)
                        x_rect = @variable(model, lower_bound = 0, upper_bound = u)
                        @constraint(model, x_rect >= x)
                        @constraint(model, x_rect <= u / (u - l) * x - u * l / (u - l))
                        set_name(x_rect, string(network_version, "x_rect_n1stab_", "layerCount", layer_counter,
                            "_neuronCount", nueron_counter, "_", neurons_names.layer, "_", neurons_names.neuron))
                        global relaxation_condition_count += 1
                        return x_rect
                    end
                end
            end
        end

        # ── advstd Technique 6: N1-gated N2/N2p triangle LP relaxation ───────
        # When --adv_std_n2_relax_threshold >= 0 and the triangle-gap-area
        # normalized measure of N1's interval at the corresponding neuron is
        # below the threshold, replace the big-M binary encoding of this
        # N2/N2p ReLU with a pure LP triangle relaxation. Sound as an
        # over-approximation: delta_relaxed >= delta_exact, and every concrete
        # feasible (x, z_N2) continues to satisfy the three triangle
        # inequalities. `_tri_gap` is the closure defined ~line 284 above.
        if adv_std_n2_relax_threshold >= 0.0 &&
           (network_version == "org" || network_version == "perturbation") &&
           !isempty(n1_neuron_bounds)
            key = (neurons_names.layer, neurons_names.neuron)
            if haskey(n1_neuron_bounds, key)
                (n1_u_val, n1_l_val) = n1_neuron_bounds[key]
                n1_score = _tri_gap(n1_l_val, n1_u_val)
                if n1_score <= adv_std_n2_relax_threshold
                    av = JuMP.all_variables(model)
                    push!(layers_info_dict,
                          (neurons_names.layer, neurons_names.neuron) => (u, l, length(av)))
                    x_rect = @variable(model)
                    set_lower_bound(x_rect, 0.0)
                    set_upper_bound(x_rect, max(0.0, u))
                    set_name(x_rect, string(network_version, "x_rect_n1relax_advstd_",
                                            "layerCount", layer_counter,
                                            "_neuronCount", nueron_counter,
                                            "_", neurons_names.layer, "_", neurons_names.neuron))
                    @constraint(model, x_rect >= 0)
                    @constraint(model, x_rect >= x)
                    @constraint(model, x_rect <= (u / (u - l)) * (x - l))
                    if network_version == "org"
                        global n_n2_relaxed_binaries_org += 1
                    else
                        global n_n2_relaxed_binaries_pert += 1
                    end
                    return x_rect
                end
            end
        end

        # ── Standard exact ReLU encoding (binary variable) ───────────────────
        av = JuMP.all_variables(model)
        push!(layers_info_dict,(neurons_names.layer,neurons_names.neuron)=>(u,l,length(av)))
        # since we know that u!=l, x is not constant, and thus x must have an associated model
        x_rect = @variable(model)
        # LP-relax N1 binaries when no_n1_binaries_and_relaxtions_only_on_n2 is active
        if no_n1_binaries_and_relaxtions_only_on_n2 && network_version == "n1_org"
            a = @variable(model)
            set_lower_bound(a, 0.0)
            set_upper_bound(a, 1.0)
        else
            a = @variable(model, binary = true)
        end
    	set_name(x_rect,string(network_version,"x_rect","_","layerCount",layer_counter,"_","neuronCount",nueron_counter,"_",string(neurons_names.layer),"_",string(neurons_names.neuron)))
    	set_name(a,string(network_version,"a","_","layerCount",layer_counter,"_","neuronCount",nueron_counter,"_",string(neurons_names.layer),"_",string(neurons_names.neuron)))
        # refined big-M formulation that takes advantage of the knowledge
        # that lower and upper bounds  are different.
        @constraint(model, x_rect <= x + (-l) * (1 - a))
        @constraint(model, x_rect >= x)
        @constraint(model, x_rect <= u * a)
        @constraint(model, x_rect >= 0)

        # ── Cross-copy linking: conditional constraints using N2(x)'s binary ──
        # Links N2(x') post-ReLU to N2(x)'s activation via perturbation bounds
        # derived through N1's zonotope. Sound: tightens LP relaxation without
        # removing any binaries. Transfer-only (standard has no second copy).
        if constrain_n2_xp_via_n1_zonotope && network_version == "n2_pert"
            m_idx = layer_counter
            k_idx = neurons_names.neuron
            # Look up N2(x)'s binary variable by name
            a_pre_name = string("n2_org", "a", "_", "layerCount", layer_counter,
                "_", "neuronCount", nueron_counter, "_",
                string(neurons_names.layer), "_", string(neurons_names.neuron))
            a_pre = variable_by_name(model, a_pre_name)
            if a_pre !== nothing &&
               !isempty(relu_n2pert_up_bounds) && m_idx >= 1 && m_idx <= length(relu_n2pert_up_bounds) &&
               k_idx >= 1 && k_idx <= length(relu_n2pert_up_bounds[m_idx]) &&
               !isempty(n2_preact_up_bounds) && m_idx <= length(n2_preact_up_bounds)
                # Perturbation interval: N2(x') - N2(x)
                u_int = relu_n2pert_up_bounds[m_idx][k_idx]
                l_int = relu_n2pert_down_bounds[m_idx][k_idx]
                # Tighter bounds via N1: composed - diff
                if !isempty(relu_comp_up_bounds) && m_idx <= length(relu_comp_up_bounds) &&
                   !isempty(relu_diff_up_bounds) && m_idx <= length(relu_diff_up_bounds)
                    u_int_n1 = relu_comp_up_bounds[m_idx][k_idx] - relu_diff_down_bounds[m_idx][k_idx]
                    l_int_n1 = relu_comp_down_bounds[m_idx][k_idx] - relu_diff_up_bounds[m_idx][k_idx]
                    u_int = min(u_int, u_int_n1)
                    l_int = max(l_int, l_int_n1)
                end
                u_pre = n2_preact_up_bounds[m_idx][k_idx]
                l_pre = n2_preact_down_bounds[m_idx][k_idx]
                # Conditional intervals: N2(x') preact given N2(x) activation
                lA = l_int;          uA = u_pre + u_int   # N2(x) active
                lI = l_pre + l_int;  uI = u_int           # N2(x) inactive
                M_val = u + (-l)
                # Active-case upper bound (a_pre=1 → tight, a_pre=0 → slack)
                if lA >= 0.0
                    @constraint(model, x_rect <= x + M_val * (1 - a_pre))
                elseif uA <= 0.0
                    @constraint(model, x_rect <= M_val * (1 - a_pre))
                elseif uA > 0.0
                    @constraint(model, x_rect <= (uA / (uA - lA)) * (x - lA) + M_val * (1 - a_pre))
                end
                # Inactive-case upper bound (a_pre=0 → tight, a_pre=1 → slack)
                if lI >= 0.0
                    @constraint(model, x_rect <= x + M_val * a_pre)
                elseif uI <= 0.0
                    @constraint(model, x_rect <= M_val * a_pre)
                elseif uI > 0.0
                    @constraint(model, x_rect <= (uI / (uI - lI)) * (x - lI) + M_val * a_pre)
                end
            end
        end

        # Manually set the bounds for x_rect so they can be used by downstream operations.
        set_lower_bound(x_rect, 0)
        set_upper_bound(x_rect, u)
        return x_rect
    end
end

@enum ReLUType split = 0 zero_output = -1 linear_in_input = 1 constant_output = 2

function get_relu_type(l::Real, u::Real)::ReLUType
    if u <= 0
        return zero_output
    elseif u == l
        return constant_output
    elseif l >= 0
        return linear_in_input
    else
        return split
    end
end

struct ReLUInfo
    lowerbounds::Array{Real}
    upperbounds::Array{Real}
end

function Base.show(io::IO, s::ReLUInfo)
    relutypes = get_relu_type.(s.lowerbounds, s.upperbounds)
    print(io, "  Behavior of ReLUs - ")
    for t in instances(ReLUType)
        n = count(x -> x == t, relutypes)
        print(io, "$t: $n")
        if t != last(instances(ReLUType))
            print(io, ", ")
        end
    end
end

"""
Calculates the lower_bound only if `u` is positive; otherwise, returns `u` (since we expect)
the ReLU to be fixed to zero anyway.
"""
function lazy_tight_lowerbound(
    x::JuMPLinearType,
    u::Real;
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
    cutoff = 0,
)::Real
    (u <= cutoff) ? u : tight_lowerbound(x; nta = nta, cutoff = cutoff)
end

function relu(x::JuMPLinearType)::JuMP.AffExpr
    u = tight_upperbound(x, cutoff = 0)
    l = lazy_tight_lowerbound(x, u, cutoff = 0)
    relu(x, l, u)
end

"""
$(SIGNATURES)
Expresses a rectified-linearity constraint: output is constrained to be equal to
`max(x, 0)`.
"""
function relu(
    x::AbstractArray{T};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
)::Array{JuMP.AffExpr} where {T<:JuMPLinearType}
    show_progress_bar::Bool =
        MIPVerify.LOGGER.levels[MIPVerify.LOGGER.level] > MIPVerify.LOGGER.levels["debug"]
    neurons_names.neuron = 0
    neurons_names.layer += 1
    global layer_counter
	global nueron_counter
    layer_counter += 1
    nueron_counter = 0
    if !show_progress_bar
        u = tight_upperbound.(x, nta = nta, cutoff = 0)
        l = lazy_tight_lowerbound.(x, u, nta = nta, cutoff = 0)
        return relu.(x, l, u)
    else
        p1 = Progress(length(x), desc = "  Calculating upper bounds: ", enabled = isinteractive())
        u = map(x_i -> (next!(p1); tight_upperbound(x_i, nta = nta, cutoff = 0)), x)
        p2 = Progress(length(x), desc = "  Calculating lower bounds: ", enabled = isinteractive())
        l = map(v -> (next!(p2); lazy_tight_lowerbound(v..., nta = nta, cutoff = 0)), zip(x, u))

        reluinfo = ReLUInfo(l, u)
        Memento.info(MIPVerify.LOGGER, "$reluinfo")

        p3 = Progress(length(x), desc = "  Imposing relu constraint: ", enabled = isinteractive())
        return x_r = map(v -> (next!(p3); relu(v...)), zip(x, l, u))
    end
end

function masked_relu(x::T, m::Real)::T where {T<:Real}
    if m < 0
        zero(T)
    elseif m > 0
        x
    else
        relu(x)
    end
end

function masked_relu(x::AbstractArray{<:Real}, m::AbstractArray{<:Real})
    masked_relu.(x, m)
end

function masked_relu(x::T, m::Real)::JuMP.AffExpr where {T<:JuMPLinearType}
    if m < 0
        zero(T)
    elseif m > 0
        x
    else
        relu(x)
    end
end

"""
$(SIGNATURES)
Expresses a masked rectified-linearity constraint, with three possibilities depending on
the value of the mask. Output is constrained to be:
```
1) max(x, 0) if m=0,
2) 0 if m<0
3) x if m>0
```
"""
function masked_relu(
    x::AbstractArray{<:JuMPLinearType},
    m::AbstractArray{<:Real};
    nta::Union{TighteningAlgorithm,Nothing} = nothing,
)::Array{JuMP.AffExpr}
    @assert(size(x) == size(m))
    s = size(m)
    # We add the constraints corresponding to the active ReLUs to the model
    zero_idx = Iterators.filter(i -> m[i] == 0, CartesianIndices(s)) |> collect
    d = Dict(zip(zero_idx, relu(x[zero_idx], nta = nta)))

    # We determine the output of the masked relu, which is either:
    #  1) the output of the relu that we have previously determined when adding the
    #     constraints to the model.
    #  2, 3) the result of applying the (elementwise) masked_relu function.
    return map(i -> m[i] == 0 ? d[i] : masked_relu(x[i], m[i]), CartesianIndices(s))
end

function maximum(xs::AbstractArray{T})::T where {T<:Real}
    return Base.maximum(xs)
end

function maximum_of_constants(xs::AbstractArray{T}) where {T<:JuMPLinearType}
    @assert all(is_constant.(xs))
    max_val = map(x -> x.constant, xs) |> maximum
    return one(JuMP.VariableRef) * max_val
end

"""
$(SIGNATURES)
Expresses a maximization constraint: output is constrained to be equal to `max(xs)`.
"""
function maximum(xs::AbstractArray{T})::JuMP.AffExpr where {T<:JuMPLinearType}
    if length(xs) == 1
        return xs[1]
    end

    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)

    # TODO (vtjeng): [PERF] skip calculating lower_bound for index if upper_bound is lower than
    # largest current lower_bound.
    p1 = Progress(length(xs), desc = "  Calculating upper bounds: ", enabled = isinteractive())
    us = map(x_i -> (next!(p1); tight_upperbound(x_i)), xs)
    p2 = Progress(length(xs), desc = "  Calculating lower bounds: ", enabled = isinteractive())
    ls = map(x_i -> (next!(p2); tight_lowerbound(x_i)), xs)

    l = Base.maximum(ls)
    u = Base.maximum(us)

    if l == u
        return one(T) * l
        Memento.info(MIPVerify.LOGGER, "Output of maximum is constant.")
    end
    # at least one index will satisfy this property because of check above.
    filtered_indexes = us .> l

    # TODO (vtjeng): Smarter log output if maximum function is being used more than once (for example, in a max-pooling layer).
    Memento.info(
        MIPVerify.LOGGER,
        "Number of inputs to maximum function possibly taking maximum value: $(filtered_indexes |> sum)",
    )

    return maximum(xs[filtered_indexes], ls[filtered_indexes], us[filtered_indexes])
end

function maximum(
    xs::AbstractArray{T,1},
    ls::AbstractArray{<:Real,1},
    us::AbstractArray{<:Real,1},
)::JuMP.AffExpr where {T<:JuMPLinearType}

    @assert length(xs) > 0
    @assert length(xs) == length(ls)
    @assert length(xs) == length(us)

    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    if length(xs) == 1
        return first(xs)
    else
        l = Base.maximum(ls)
        u = Base.maximum(us)
        x_max = @variable(model, lower_bound = l, upper_bound = u)
        a = @variable(model, [1:length(xs)], binary = true)
        @constraint(model, sum(a) == 1)
        for (i, x) in enumerate(xs)
            umaxi = Base.maximum(us[1:end.!=i])
            @constraint(model, x_max <= x + (1 - a[i]) * (umaxi - ls[i]))
            @constraint(model, x_max >= x)
        end
        return x_max
    end
end

"""
$(SIGNATURES)
Expresses a one-sided maximization constraint: output is constrained to be at least
`max(xs)`.

Only use when you are minimizing over the output in the objective.

NB: If all of xs are constant, we simply return the largest of them.
"""
function maximum_ge(xs::AbstractArray{T})::JuMPLinearType where {T<:JuMPLinearType}
    @assert length(xs) > 0
    if all(is_constant.(xs))
        return maximum_of_constants(xs)
    end
    if length(xs) == 1
        return first(xs)
    end
    # at least one of xs is not constant.
    model = owner_model(xs)
    x_max = @variable(model)
    @constraint(model, x_max .>= xs)
    return x_max
end

"""
$(SIGNATURES)
Expresses a one-sided absolute-value constraint: output is constrained to be at least as
large as `|x|`.

Only use when you are minimizing over the output in the objective.
"""
function abs_ge(x::JuMPLinearType)::JuMP.AffExpr
    if is_constant(x)
        return one(JuMP.VariableRef) * abs(x.constant)
    end
    model = owner_model(x)
    u = upper_bound(x)
    l = lower_bound(x)
    if u <= 0
        return -x
    elseif l >= 0
        return x
    else
        x_abs = @variable(model)
        @constraint(model, x_abs >= x)
        @constraint(model, x_abs >= -x)
        set_lower_bound(x_abs, 0)
        set_upper_bound(x_abs, max(-l, u))
        return x_abs
    end
end

function get_target_indexes(
    target_index::Integer,
    array_length::Integer;
    invert_target_selection::Bool = false,
)

    get_target_indexes(
        [target_index],
        array_length,
        invert_target_selection = invert_target_selection,
    )

end

function get_target_indexes(
    target_indexes::Array{<:Integer,1},
    array_length::Integer;
    invert_target_selection::Bool = false,
)::AbstractArray{<:Integer,1}

    @assert length(target_indexes) >= 1
    @assert all(target_indexes .>= 1) && all(target_indexes .<= array_length)

    invert_target_selection ? filter((x) -> x ∉ target_indexes, 1:array_length) : target_indexes
end

function get_vars_for_max_index(
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1},
)::Tuple{JuMPLinearType,Array{<:JuMPLinearType,1}}

    @assert length(xs) >= 1

    target_vars = xs[Bool[i ∈ target_indexes for i in 1:length(xs)]]
    nontarget_vars = xs[Bool[i ∉ target_indexes for i in 1:length(xs)]]

    maximum_target_var = length(target_vars) == 1 ? target_vars[1] : MIPVerify.maximum(target_vars)

    return (maximum_target_var, nontarget_vars)
end

"""
$(SIGNATURES)

Imposes constraints ensuring that one of the elements at the target_indexes is the
largest element of the array x. More specifically, we require `x[j] - x[i] ≥ margin` for
some `j ∈ target_indexes` and for all `i ∉ target_indexes`.
"""
function set_max_indexes(
    model::Model,
    xs::Array{<:JuMPLinearType,1},
    target_indexes::Array{<:Integer,1};
    margin::Real = 0.001,
)::Nothing

    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(xs, target_indexes)

    @constraint(model, nontarget_vars .<= maximum_target_var - margin)
    return nothing
end
