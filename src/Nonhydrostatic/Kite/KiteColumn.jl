# The vertical part of the kite model: the Galerkin discretization of the vertical terms in every
# column, ρ, ρθ in DG0 (layers) and w in CG1 (interfaces), all inner products in the vertical with the
# trapezoidal rule of every layer (the Gauss–Lobatto rule of CG1, as the Radau rule horizontally), so
# that the vertical mass of w is diagonal and the columns at the Radau nodes are independent.
#
# With the trapezoidal rule the projection of a DG0 field ρ times a CG1 field w into CG1 is ρ̄_j w_j at
# interface j, with the thickness-weighted mean ρ̄_j = (dz_{j-1} ρ_{j-1} + dz_j ρ_j) / (dz_{j-1} + dz_j)
# (`interface_mean`), and the mass of w at interface j is dzh_j. The terms per column (horizontal
# mass M_h,ii cancels):
#
#   mass flux        F_j = ρ̄_j w_j                         (Π_CG1(ρ w))
#   ρθ flux          G_j = θ̄_j F_j (+ the upwind penalty)  (Π_CG1(θ F))
#   continuity       dz_k ρ̇_k = -(F_{k+1} - F_k),   dz_k (ρθ)˙_k = -(G_{k+1} - G_k)
#   w                dzh_j ẇ_j = -c_p θ̄_j (π_j - π_{j-1}) - g dzh_j - dzh_j r_j w_j
#
# The pressure gradient is θ̄ times the adjoint of the divergence, so with the kinetic energy
# Σ_j dzh_j ρ̄_j w_j²/2 (the same rule) the exchange with the internal and potential energy is exact:
# Σ_j dzh_j F_j ẇ_j + Σ_k dz_k (c_p π_k (ρθ)˙_k + g z_k ρ̇_k) = 0 (`vertical_tendency!`).

"Weight of the layer below in the interface mean of the trapezoidal rule, `dz_{j-1} / (dz_{j-1} + dz_j)`."
function interface_weights(v::VerticalGrid)
    n = num_layers(v)
    a = zeros(n + 1)
    for j in 2:n
        a[j] = v.dz[j - 1] / (v.dz[j - 1] + v.dz[j])
    end
    return a
end

@inline interface_mean(x, a, i, j) = a[j] * x[j - 1, i] + (1 - a[j]) * x[j, i]

"Exner pressure `π0` of the base state and `γ0 = κ π0 / ρθ0`, its derivative by `ρθ`."
@kernel function kite_exner_kernel!(π0, γ0, @Const(ρθ0))
    k, i = @index(Global, NTuple)
    @inbounds begin
        x = ρθ0[k, i]
        π = (Rd * x / P0)^(Rd / Cvd)
        π0[k, i] = π
        γ0[k, i] = Rd / Cvd * π / x
    end
end

"""
Upwind penalty of the ρθ flux at the interfaces of the stage state,
`P_j = α |F_j| (θ_{j-1} - θ_j) / 2` with `F_j = ρ̄_j w_j` (zero at the ground and
the top): the central flux `θ̄ F` plus `P` is the upwind flux of DG0 for `α = 1`.
"""
@kernel function kite_theta_penalty_kernel!(P, @Const(θs), @Const(ρs), @Const(ws), @Const(a), α, nlev)
    i = @index(Global)
    @inbounds begin
        P[1, i] = P[nlev + 1, i] = 0.0
        for j in 2:nlev
            P[j, i] = α * abs(interface_mean(ρs, a, i, j) * ws[j, i]) * (θs[j - 1, i] - θs[j, i]) / 2
        end
    end
end

"""
Implicit step of one column: `w`, `ρ`, `ρθ` at `t + τ` from the base state
(`w0`, `ρθ0`, `π0`, `γ0`), the explicit parts `tw`, `ρexpl`, `ρθexpl` and the
interface means `ρ̄`, `θ̄` of the stage state (`ρs`, `θs`): the mass flux
`F_j = ρ̄_j (β w_j + (1 - β) w0_j)`, the ρθ flux `θ̄_j F_j + P_j`, `π` linearized
around `π0`, `π = π0 + γ0 (ρθ - ρθ0)`, the pressure gradient off-centred with
`β`, and Klemp's damping `r_j w_j` implicit; one tridiagonal system in `w`.
"""
@kernel function kite_column_kernel!(w, ρ, ρθ, cp_, dp_, @Const(w0), @Const(ρθ0), @Const(π0), @Const(γ0),
                                     @Const(θs), @Const(ρs), @Const(P), @Const(tw), @Const(ρexpl), @Const(ρθexpl),
                                     @Const(dz), @Const(dzh), @Const(a), @Const(rayleigh), τ, β, nlev)
    i = @index(Global)
    @inbounds begin
        θ̄(j) = interface_mean(θs, a, i, j)
        # G_j = B_j w_j + C_j (zero at the ground and the top)
        B(j) = 2 <= j <= nlev ? θ̄(j) * interface_mean(ρs, a, i, j) * β : 0.0
        C(j) = 2 <= j <= nlev ? θ̄(j) * interface_mean(ρs, a, i, j) * (1 - β) * w0[j, i] + P[j, i] : 0.0
        # layer k: γ = γ0 τ/dz and the explicit π
        γl = γ0[1, i] * τ / dz[1]
        π0l = π0[1, i]
        πl = π0l + γ0[1, i] * (ρθexpl[1, i] - ρθ0[1, i])
        Bl, Cl, Bc, Cc = 0.0, 0.0, B(2), C(2)
        cp_[1, i] = dp_[1, i] = 0.0
        for j in 2:nlev
            γu = γ0[j, i] * τ / dz[j]
            π0u = π0[j, i]
            πu = π0u + γ0[j, i] * (ρθexpl[j, i] - ρθ0[j, i])
            Bu, Cu = B(j + 1), C(j + 1)
            θj = θ̄(j)
            aj = τ * Cpd * θj * β / dzh[j]
            lower = -aj * γl * Bl
            diag = 1 + τ * rayleigh[j] + aj * (γu + γl) * Bc
            upper = -aj * γu * Bu
            rhs = w0[j, i] + τ * (tw[j, i] - GRAV) - τ * Cpd * θj * (1 - β) * (π0u - π0l) / dzh[j] -
                  aj * (πu - πl) - aj * (-γu * Cu + (γu + γl) * Cc - γl * Cl)
            denom = diag - lower * cp_[j - 1, i]
            cp_[j, i] = upper / denom
            dp_[j, i] = (rhs - lower * dp_[j - 1, i]) / denom
            γl, π0l, πl = γu, π0u, πu
            Bl, Cl, Bc, Cc = Bc, Cc, Bu, Cu
        end
        w[nlev + 1, i] = 0.0
        for j in nlev:-1:2
            w[j, i] = dp_[j, i] - cp_[j, i] * w[j + 1, i]
        end
        w[1, i] = 0.0
        Flow = Glow = 0.0
        for k in 1:nlev
            F = G = 0.0
            if k < nlev
                j = k + 1
                F = interface_mean(ρs, a, i, j) * (β * w[j, i] + (1 - β) * w0[j, i])
                G = θ̄(j) * F + P[j, i]
            end
            ρ[k, i] = ρexpl[k, i] - τ / dz[k] * (F - Flow)
            ρθ[k, i] = ρθexpl[k, i] - τ / dz[k] * (G - Glow)
            Flow, Glow = F, G
        end
    end
end

"""
Vertical tracer transport of one column: `ρq = ρq_expl - τ (Q_{k+1} - Q_k)/dz_k`
with `Q_j = q̄_j F_j` and the mass flux `F_j` of the column step, plus the upwind
penalty `α |ρ̄_j ws_j| (q_{j-1} - q_j)/2` of the stage state as for ρθ; for
`q = 1` the update of `ρ`.
"""
@kernel function kite_tracer_column_kernel!(ρq, @Const(ρqexpl), @Const(q), @Const(ρs), @Const(ws), @Const(w), @Const(w0),
                                            @Const(dz), @Const(a), α, τ, β, nlev)
    i = @index(Global)
    @inbounds begin
        Qlow = 0.0
        for k in 1:nlev
            Q = 0.0
            if k < nlev
                j = k + 1
                ρj = interface_mean(ρs, a, i, j)
                Q = interface_mean(q, a, i, j) * ρj * (β * w[j, i] + (1 - β) * w0[j, i]) +
                    α * abs(ρj * ws[j, i]) * (q[j - 1, i] - q[j, i]) / 2
            end
            ρq[k, i] = ρqexpl[k, i] - τ / dz[k] * (Q - Qlow)
            Qlow = Q
        end
    end
end

"""
    vertical_tendency!(dw, dρ, dρθ, m::KiteNonhydrostatic, y)

The semi-discrete vertical terms of the state `y` in every column (the operator
that the column step integrates implicitly, without the upwind penalty and the
damping): `dzh_j ẇ_j = -c_p θ̄_j (π_j - π_{j-1}) - g dzh_j`,
`dz_k ρ̇_k = -(F_{k+1} - F_k)`, `dz_k (ρθ)˙_k = -(θ̄_{k+1} F_{k+1} - θ̄_k F_k)`
with `F_j = ρ̄_j w_j`.
"""
function vertical_tendency!(dw, dρ, dρθ, m::KiteNonhydrostatic, y)
    a, v, nlev = Array(m.below), m.vgrid, num_layers(m)
    ρ, ρθ, w = Array(y.ρ), Array(y.ρθ), Array(y.w)
    θ = ρθ ./ ρ
    π = @. (Rd * ρθ / P0)^(Rd / Cvd)
    F, G = zeros(size(w)), zeros(size(w))
    W = zeros(size(w))
    for i in axes(w, 2), j in 2:nlev
        F[j, i] = interface_mean(ρ, a, i, j) * w[j, i]
        G[j, i] = interface_mean(θ, a, i, j) * F[j, i]
        W[j, i] = -Cpd * interface_mean(θ, a, i, j) * (π[j, i] - π[j - 1, i]) / v.dzh[j] - GRAV
    end
    copyto!(dw, W)
    copyto!(dρ, -(F[2:end, :] .- F[1:end - 1, :]) ./ v.dz)
    copyto!(dρθ, -(G[2:end, :] .- G[1:end - 1, :]) ./ v.dz)
    return dw, dρ, dρθ
end

"""
    kite_balanced_column(profile, v) -> (ρ, ρθ)

Layer values of a column at rest in the discrete hydrostatic balance of the
column step, `c_p θ̄_j (π_j - π_{j-1}) = -g dzh_j` with the interface mean `θ̄`,
from the temperature of `profile(z) = (T, p, ...)` at the layer centres and its
pressure at the lowest one.
"""
function kite_balanced_column(profile, v::VerticalGrid)
    nlev, κ, a = num_layers(v), Rd / Cpd, interface_weights(v)
    T = [profile(z)[1] for z in v.z_full]
    πk = [(profile(z)[2] / P0)^κ for z in v.z_full]
    for j in 2:nlev
        πj = πk[j]
        for _ in 1:50
            θ̄ = a[j] * T[j - 1] / πk[j - 1] + (1 - a[j]) * T[j] / πj
            πj = πk[j - 1] - GRAV * v.dzh[j] / (Cpd * θ̄)
        end
        πk[j] = πj
    end
    ρθ = @. P0 / Rd * πk^(Cvd / Rd)
    return ρθ ./ (T ./ πk), ρθ
end
