# Parts shared by the nonhydrostatic dycores: physical constants, the kernel launcher with the
# blocks of points × levels, and the vertically implicit column solver (θ at the interfaces,
# the vertical mass of w, the Exner pressure of the base state, the tridiagonal solve for w, ρ, ρθ).
# One work item per column; the column index runs over the cells of the C-grid models and over the
# Radau nodes of the kite model.

"""
Nonhydrostatic models on an extruded C grid (`TRiSKNonhydrostatic`,
`ICONNonhydrostatic`): state `(vn, w, ρ, ρθ)` with `vn` at the edges × layers,
`w` at the cells × interfaces, `ρ`, `ρθ` at the cells × layers, and fields
`grid`, `vgrid`, `cell_areas`, `dz`, `vertical_theta`, `backend`.
"""
abstract type CGridNonhydrostatic end

const Rd = 287.0
const Cpd = 1004.0
const Cvd = Cpd - Rd
const P0 = 1.0e5
const GRAV = 9.80616

# Work items over (points, levels) run in blocks of 32 points × 8 levels: the neighbour tables and the
# lines of the gathered neighbours are shared by eight levels in the cache (ICON's nproma blocks).
const BLOCK = (32, 8)

function launch!(m, kernel, ndrange, args...)   # a model with a `backend`
    k = ndrange isa Tuple{Integer, Integer} ? kernel(m.backend, BLOCK) : kernel(m.backend)
    k(args...; ndrange)
    KernelAbstractions.synchronize(m.backend)
    return nothing
end

"""
Weights of the cells `j-2, j-1, j, j+1` in `θ` at interface `j` for the mass
flux `M`: linear interpolation plus `1/6` of the curvature at the upwind cell,
third-order upwind (ICON-IAP `impl_vert_adv_scalar_5diag`), linear next to the
boundaries where the upwind stencil would leave the column.
"""
@inline function theta_stencil(j, M, dz, dzh, wlow, nlev)
    σ = 1 / 6
    a, b, c, d = 0.0, wlow[j], 1 - wlow[j], 0.0
    if M >= 0
        if j >= 3
            a -= σ * dzh[j] / dz[j - 1]
            b += σ * (dzh[j] + dzh[j - 1]) / dz[j - 1]
            c -= σ * dzh[j - 1] / dz[j - 1]
        end
    elseif j <= nlev - 1
        b -= σ * dzh[j + 1] / dz[j]
        c += σ * (dzh[j] + dzh[j + 1]) / dz[j]
        d -= σ * dzh[j] / dz[j]
    end
    return (a, b, c, d)
end

"""
`θ` at the interfaces for the vertical solver. `upwind`: implicit vertical
advection of `θ` of the base state with the mass flux of the stage state over
`τ/2` (Crank–Nicolson, one pentadiagonal system per column) and the
third-order upwind interface values of the result, as ICON-IAP
(`ltheta_up_vert`); otherwise `θ` of the stage state linearly interpolated.
"""
@kernel function theta_interface_kernel!(θi, A, θm, @Const(θs), @Const(ρ0), @Const(ρθ0), @Const(ρs), @Const(ws),
                                         @Const(dz), @Const(dzh), @Const(wlow), τ, nlev, upwind)
    i = @index(Global)
    @inbounds begin
        θi[i, 1] = θi[i, nlev + 1] = 0.0
        if !upwind
            for j in 2:nlev
                θi[i, j] = wlow[j] * θs[i, j - 1] + (1 - wlow[j]) * θs[i, j]
            end
        else
            M(j) = (wlow[j] * ρs[i, j - 1] + (1 - wlow[j]) * ρs[i, j]) * ws[i, j]
            # row k: θm_k + τ/(2ρ dz) [M_{k+1} (θ_{k+1} - θm_k) - M_k (θ_k - θm_k)] = θ0_k with the interface
            # values θ_j of the upwind stencil, over the cells k-2..k+2; built when it is eliminated, from
            # the stencils of the interfaces k and k + 1 (carried along). Gaussian elimination without
            # pivoting keeps the normalized upper entries A[i, k, 1:2] and the right-hand side θm[i, k]
            # for the back substitution
            Ml, cl = 0.0, (0.0, 0.0, 0.0, 0.0)
            u1l = u2l = yl = u1ll = u2ll = yll = 0.0
            for k in 1:nlev
                Mu = k < nlev ? M(k + 1) : 0.0
                cu = k < nlev ? theta_stencil(k + 1, Mu, dz, dzh, wlow, nlev) : (0.0, 0.0, 0.0, 0.0)
                f = τ / (2 * ρs[i, k] * dz[k])
                u, l = f * Mu, f * Ml
                r1 = -l * cl[1]
                r2 = u * cu[1] - l * cl[2]
                r3 = 1 + u * (cu[2] - 1) - l * (cl[3] - 1)
                r4 = u * cu[3] - l * cl[4]
                r5 = u * cu[4]
                y = ρθ0[i, k] / ρ0[i, k]
                r2 -= r1 * u1ll
                r3 -= r1 * u2ll
                y -= r1 * yll
                r3 -= r2 * u1l
                r4 -= r2 * u2l
                y -= r2 * yl
                u1, u2, y = r4 / r3, r5 / r3, y / r3
                A[i, k, 1], A[i, k, 2], θm[i, k] = u1, u2, y
                u1ll, u2ll, yll, u1l, u2l, yl = u1l, u2l, yl, u1, u2, y
                Ml, cl = Mu, cu
            end
            θ1 = θ2 = 0.0
            for k in nlev:-1:1
                θk = θm[i, k] - A[i, k, 1] * θ1 - A[i, k, 2] * θ2
                θm[i, k] = θk
                θ1, θ2 = θk, θ1
            end
            for j in 2:nlev
                c = theta_stencil(j, M(j), dz, dzh, wlow, nlev)
                s = 0.0
                for n in 1:4
                    cell = j - 3 + n
                    1 <= cell <= nlev && (s += c[n] * θm[i, cell])
                end
                θi[i, j] = s
            end
        end
    end
end

"""
    vertical_mass(v, kind) -> 3 × (nlev + 1)

Rows of the vertical mass matrix of `w` divided by `dzh` for
`vertical_solve_kernel!`: `:lumped` (identity) or `:consistent` (CG1 hat
functions, tridiagonal, rows summing to one).
"""
function vertical_mass(v::VerticalGrid, kind::Symbol)
    n = num_layers(v)
    M = zeros(3, n + 1)
    M[2, :] .= 1
    kind == :lumped && return M
    kind == :consistent || throw(ArgumentError("vertical_mass must be :consistent or :lumped"))
    for j in 2:n
        M[:, j] .= (v.dz[j - 1], 2 * (v.dz[j - 1] + v.dz[j]), v.dz[j]) ./ (6 * v.dzh[j])
    end
    return M
end

"Exner pressure `π0` of the base state and `γ0 = κ π0 / ρθ0`, its derivative by `ρθ`, once per step."
@kernel function base_exner_kernel!(π0, γ0, @Const(ρθ0))
    i, k = @index(Global, NTuple)
    @inbounds begin
        x = ρθ0[i, k]
        π = (Rd * x / P0)^(Rd / Cvd)
        π0[i, k] = π
        γ0[i, k] = Rd / Cvd * π / x
    end
end

"""
Vertically implicit step of one column: `w`, `ρ`, `ρθ` at `t + τ` from the
explicit parts, with `θ` at the interfaces from `theta_interface_kernel!` in
the `ρθ` flux (and with `iap_theta` in the pressure gradient, else `θ` of the
stage state linearly interpolated), `ρ` there from the stage state and the
linearization of `π` around the base state (`π0`, `γ0` of
`base_exner_kernel!`). `wmass[:, j]` is row `j` of the vertical mass matrix of
`w` divided by `dzh[j]` (lower, diagonal, upper): `(0, 1, 0)` lumped, pointwise
at the interfaces; `(dz[j-1], 2(dz[j-1] + dz[j]), dz[j]) / 6dzh[j]` for the
consistent CG1 mass, which keeps the system tridiagonal. The coefficients of
the cells and interfaces next to interface `j` are carried along the sweep.
"""
@kernel function vertical_solve_kernel!(w, ρ, ρθ, cp_, dp_, @Const(w0), @Const(ρθ0), @Const(π0), @Const(γ0),
                                        @Const(θs), @Const(θi), @Const(ρs), @Const(tw), @Const(ρexpl),
                                        @Const(ρθexpl), @Const(dz), @Const(dzh), @Const(wlow), @Const(rayleigh),
                                        @Const(wmass), τ, β, nlev, iap_theta)
    i = @index(Global)
    @inbounds begin
        θh(j) = iap_theta ? θi[i, j] : wlow[j] * θs[i, j - 1] + (1 - wlow[j]) * θs[i, j]
        ρh(j) = wlow[j] * ρs[i, j - 1] + (1 - wlow[j]) * ρs[i, j]
        # interface j couples to j - 1 and j + 1 through M = b w + c (zero at the boundaries), in the ρθ flux
        # with θ_j: B = θ_j b, C = θ_j c
        B(j) = 2 <= j <= nlev ? θi[i, j] * ρh(j) * β : 0.0
        C(j) = 2 <= j <= nlev ? θi[i, j] * ρh(j) * (1 - β) * w0[i, j] : 0.0
        # cell k: γ = κ π0/ρθ0 τ/dz and the explicit π linearized around π0
        γl = γ0[i, 1] * τ / dz[1]
        π0l = π0[i, 1]
        πl = π0l + γ0[i, 1] * (ρθexpl[i, 1] - ρθ0[i, 1])
        Bl, Cl, Bc, Cc = 0.0, 0.0, B(2), C(2)
        # Thomas algorithm, forward sweep over j = 2:nlev
        cp_[i, 1] = 0.0
        dp_[i, 1] = 0.0
        for j in 2:nlev
            g = γ0[i, j]
            γu = g * τ / dz[j]
            π0u = π0[i, j]
            πu = π0u + g * (ρθexpl[i, j] - ρθ0[i, j])
            Bu, Cu = B(j + 1), C(j + 1)
            θj = θh(j)
            aj = τ * Cpd * θj * β / dzh[j]
            ml, md, mu = wmass[1, j], wmass[2, j], wmass[3, j]
            lower = ml - aj * γl * Bl
            diag = md + τ * rayleigh[j] + aj * (γu + γl) * Bc
            upper = mu - aj * γu * Bu
            rhs = ml * w0[i, j - 1] + md * w0[i, j] + mu * w0[i, j + 1] + τ * (tw[i, j] - GRAV) - τ * Cpd * θj * (1 - β) * (π0u - π0l) / dzh[j] -
                  aj * (πu - πl) - aj * (-γu * Cu + (γu + γl) * Cc - γl * Cl)
            denom = diag - lower * cp_[i, j - 1]
            cp_[i, j] = upper / denom
            dp_[i, j] = (rhs - lower * dp_[i, j - 1]) / denom
            γl, π0l, πl = γu, π0u, πu
            Bl, Cl, Bc, Cc = Bc, Cc, Bu, Cu
        end
        w[i, nlev + 1] = 0.0
        for j in nlev:-1:2
            w[i, j] = dp_[i, j] - cp_[i, j] * w[i, j + 1]
        end
        w[i, 1] = 0.0
        # mass and ρθ fluxes M_j, θ_j M_j and the new ρ, ρθ
        Mlow = θMlow = 0.0
        for k in 1:nlev
            if k < nlev
                ρj = ρh(k + 1)
                M = ρj * (β * w[i, k + 1] + (1 - β) * w0[i, k + 1])
                θM = θi[i, k + 1] * M
            else
                M = θM = 0.0
            end
            ρ[i, k] = ρexpl[i, k] - τ / dz[k] * (M - Mlow)
            ρθ[i, k] = ρθexpl[i, k] - τ / dz[k] * (θM - θMlow)
            Mlow, θMlow = M, θM
        end
    end
end

