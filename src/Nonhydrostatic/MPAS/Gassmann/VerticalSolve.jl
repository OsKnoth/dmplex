# The pentadiagonal vertical solve of ICON-IAP (Gassmann and Herzog 2008, Gassmann 2013) for the
# vertical mass flux, with the vertical kinetic energy implicit.

"""
`w` at interface `j` as a linear function of the mass fluxes, `w_j = c₋ m_{j-1} + c₀ m_j + c₊ m_{j+1} + c`:
`w_j = (2m_j - ρ̄_j w⁰_j)/ρ⁰_j` with `ρ̄_j` the new density interpolated to the interface,
`ρ_k = ρexpl_k - τ (m_{k+1} - m_k)/dz_k`. Zero at the ground and the top.
"""
@inline function w_of_mass_flux(i, j, w0, ρ0, ρexpl, dz, wlow, τ, nlev)
    (2 <= j <= nlev) || return (0.0, 0.0, 0.0, 0.0)
    @inbounds begin
        R = wlow[j] * ρ0[i, j - 1] + (1 - wlow[j]) * ρ0[i, j]
        ρe = wlow[j] * ρexpl[i, j - 1] + (1 - wlow[j]) * ρexpl[i, j]
        s = w0[i, j] / R
        gl, gu = τ * wlow[j] / dz[j - 1], τ * (1 - wlow[j]) / dz[j]
        return (-s * gl, 2 / R + s * (gl - gu), s * gu, -s * ρe)
    end
end

"`c` plus `s` times the coefficients of `w` at the interface `j + o` (`w_of_mass_flux`), in row `j`."
@inline add_w(c, s, o, Wn) = ntuple(n -> n == 6 ? c[6] - s * Wn[4] :
                                         1 <= n - o - 1 <= 3 ? c[n] + s * Wn[n - o - 1] : c[n], 6)

"""
Vertically implicit step of one column as in ICON-IAP (`divergent_modes_5band`): the unknown is
the mass flux at the interfaces, time-centred in the products, `m = (ρ̄ w⁰ + ρ⁰ w)/2` with the new
density `ρ̄` and `ρ⁰`, `w⁰` of the old state. Per interface

    w - w⁰ = τ (tw - g) - τ c_p θ_j [(1 - β) ∂π⁰/∂z + β ∂π/∂z]/dzh - τ ∂K_w/∂z - τ r w

with `π` linearized around `π⁰` (`base_exner_kernel!`), `ρθ_k = ρθexpl_k - τ (θ_{k+1} m_{k+1} -
θ_k m_k)/dz_k`, and the vertical kinetic energy implicit, `w⁰ w/2` at the interfaces averaged to the
cells as `w_kinetic_kernel!` (`tw` without it). `w` through `w_of_mass_flux` couples three mass
fluxes, `∂K_w/∂z` then five: one pentadiagonal system per column, Gaussian elimination without
pivoting, the factors in `A[i, j, 1:4]` (diagonal, two upper entries, right-hand side, then `m`).
ICON-IAP takes `β = c_v/c_p`.
"""
@kernel function five_band_solve_kernel!(w, ρ, ρθ, A, @Const(w0), @Const(ρ0), @Const(π0), @Const(γ0),
                                         @Const(θs), @Const(θi), @Const(tw), @Const(ρexpl), @Const(ρθexpl),
                                         @Const(ρθ0), @Const(dz), @Const(dzh), @Const(wlow), @Const(rayleigh), τ, β,
                                         nlev, iap_theta)
    i = @index(Global)
    @inbounds begin
        W(j) = w_of_mass_flux(i, j, w0, ρ0, ρexpl, dz, wlow, τ, nlev)
        πx(k) = π0[i, k] + γ0[i, k] * (ρθexpl[i, k] - ρθ0[i, k])
        Ul = Vl = El = Ql = 0.0
        Ull = Vll = Ell = Qll = 0.0
        for j in 2:nlev
            # row j over m_{j-2..j+2} and the right-hand side: c[1:5], c[6]
            c = (0.0, 0.0, 0.0, 0.0, 0.0, w0[i, j] + τ * (tw[i, j] - GRAV))
            Wj = W(j)
            c = add_w(c, 1 + τ * rayleigh[j], 0, Wj)
            # pressure gradient: π_k = πx_k - γ_k τ/dz_k (θ_{k+1} m_{k+1} - θ_k m_k)
            θj = iap_theta ? θi[i, j] : wlow[j] * θs[i, j - 1] + (1 - wlow[j]) * θs[i, j]
            aj = τ * Cpd * θj * β / dzh[j]
            gu, gl = γ0[i, j] * τ / dz[j], γ0[i, j - 1] * τ / dz[j - 1]
            c = (c[1], c[2] - aj * gl * θi[i, j - 1], c[3] + aj * (gu + gl) * θi[i, j], c[4] - aj * gu * θi[i, j + 1],
                 c[5], c[6] - aj * (πx(j) - πx(j - 1)) - τ * Cpd * θj * (1 - β) * (π0[i, j] - π0[i, j - 1]) / dzh[j])
            # τ ∂K_w/∂z with K_k = (dzh_k w⁰_k w_k + dzh_{k+1} w⁰_{k+1} w_{k+1})/4dz_k
            f = τ / (4 * dzh[j])
            c = add_w(c, f * dzh[j] * w0[i, j] * (1 / dz[j] - 1 / dz[j - 1]), 0, Wj)
            c = add_w(c, f * dzh[j + 1] * w0[i, j + 1] / dz[j], 1, W(j + 1))
            c = add_w(c, -f * dzh[j - 1] * w0[i, j - 1] / dz[j - 1], -1, W(j - 1))
            c1, c2, c3, c4, c5, r = c
            # elimination with the rows j - 2 and j - 1
            if j > 3
                k = c1 / Ull
                c2 -= k * Vll; c3 -= k * Ell; r -= k * Qll
            end
            if j > 2
                l = c2 / Ul
                c3 -= l * Vl; c4 -= l * El; r -= l * Ql
            end
            A[i, j, 1], A[i, j, 2], A[i, j, 3], A[i, j, 4] = c3, c4, c5, r
            Ull, Vll, Ell, Qll = Ul, Vl, El, Ql
            Ul, Vl, El, Ql = c3, c4, c5, r
        end
        m1 = m2 = 0.0
        for j in nlev:-1:2
            mj = (A[i, j, 4] - A[i, j, 2] * m1 - A[i, j, 3] * m2) / A[i, j, 1]
            A[i, j, 4] = mj
            m1, m2 = mj, m1
        end
        A[i, 1, 4] = A[i, nlev + 1, 4] = 0.0
        w[i, 1] = w[i, nlev + 1] = 0.0
        for j in 2:nlev
            c = W(j)
            w[i, j] = c[1] * A[i, j - 1, 4] + c[2] * A[i, j, 4] + c[3] * A[i, j + 1, 4] + c[4]
        end
        for k in 1:nlev
            ml, mu = A[i, k, 4], A[i, k + 1, 4]
            ρ[i, k] = ρexpl[i, k] - τ / dz[k] * (mu - ml)
            ρθ[i, k] = ρθexpl[i, k] - τ / dz[k] * (θi[i, k + 1] * mu - θi[i, k] * ml)
        end
    end
end
