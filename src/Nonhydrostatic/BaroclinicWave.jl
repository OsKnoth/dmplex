"""
    baroclinic_wave(lon, lat, z; radius, omega, perturbation=true) -> (T, p, u, v)

Temperature, pressure and wind of the dry baroclinic wave of Ullrich, Melvin,
Jablonowski and Staniforth (2014), shallow atmosphere, in height coordinates
(as `BaroWaveDryExample` in CGDycore.jl), with the stream-function
perturbation of DCMIP 2016 centred at (20°E, 40°N).
"""
function baroclinic_wave(lon, lat, z; radius=6.37122e6, omega=7.292e-5, perturbation::Bool=true)
    T0E, T0P, B, K, Γ = 310.0, 240.0, 2.0, 3.0, 0.005
    T0 = (T0E + T0P) / 2
    A = 1 / Γ
    Bc = (T0 - T0P) / (T0 * T0P)
    C = (K + 2) / 2 * (T0E - T0P) / (T0E * T0P)
    H = Rd * T0 / GRAV
    ζ = z / (B * H)
    τ1 = A * Γ / T0 * exp(Γ / T0 * z) + Bc * (1 - 2ζ^2) * exp(-ζ^2)
    τ2 = C * (1 - 2ζ^2) * exp(-ζ^2)
    I1 = A * (exp(Γ / T0 * z) - 1) + Bc * z * exp(-ζ^2)
    I2 = C * z * exp(-ζ^2)
    interior = cos(lat)^K - K / (K + 2) * cos(lat)^(K + 2)
    T = 1 / (τ1 - τ2 * interior)
    p = P0 * exp(-GRAV / Rd * (I1 - I2 * interior))
    U = GRAV / radius * K * I2 * (cos(lat)^(K - 1) - cos(lat)^(K + 1)) * T
    rc = radius * cos(lat)
    u = -omega * rc + sqrt((omega * rc)^2 + rc * U)
    v = 0.0
    if perturbation
        du, dv = baroclinic_perturbation(lon, lat, z)
        u += du
        v += dv
    end
    return T, p, u, v
end

"""
    thermal_wind(lon, lat, z; T0=300, epsilon=0.1, zs=12000, ps=1e5, radius, omega) -> (T, p, u, v)

A steady rotating wind in thermal-wind balance, shallow atmosphere, height
coordinates. The temperature falls towards the poles near the ground and the
meridional gradient decays with height,

    1/T = (1 + ε sin²φ s(z)) / T0,  s(z) = sech²(z/zs),  S(z) = ∫₀ᶻ s = zs tanh(z/zs),

the pressure is hydrostatic with the same surface pressure everywhere (no
wind at the ground), `p = ps exp(-g (z + ε sin²φ S(z)) / (R T0))`, and the
zonal wind follows from the gradient-wind balance
`(2Ω sin φ + u tan φ / a) u = -(R T / a) ∂ln p/∂φ`:

    u = -Ω r + √((Ω r)² + 2 g ε cos²φ S(z) T / T0),  r = a cos φ.

The westerly grows up to about `zs` and is constant above (about 25 m/s at
the equator for `ε = 0.1`), so the upper levels and the damping layer see
no shear. It is an exact steady state of the dry Euler equations, stably
stratified; the shear below `zs` is weakly baroclinically unstable.
"""
function thermal_wind(lon, lat, z; T0=300.0, epsilon=0.1, zs=12000.0, ps=1e5, radius=6.37122e6, omega=7.292e-5)
    s2 = sin(lat)^2
    S = zs * tanh(z / zs)
    T = T0 / (1 + epsilon * s2 * sech(z / zs)^2)
    p = ps * exp(-GRAV * (z + epsilon * s2 * S) / (Rd * T0))
    rc = radius * cos(lat)
    u = -omega * rc + sqrt((omega * rc)^2 + 2GRAV * epsilon * cos(lat)^2 * S * T / T0)
    return T, p, u, 0.0
end

"""
    baroclinic_channel(x, y, z; Ly=6e6, f0=1.0313e-4, perturbation=true) -> (T, p, u, v)

The baroclinic wave in a Cartesian f-plane channel of Ullrich, Reed and
Jablonowski (2015), flat ground, in height coordinates, for a doubly periodic
box `[0, Lx) × [0, 2Ly)`: the channel `0 ≤ y ≤ Ly` and its mirror image
`Ly ≤ y ≤ 2Ly` with the easterly jet. The jet vanishes with its shear at
`y = 0` and `Ly`, so the mirrored geopotential is smooth and periodic, and
`v = 0` on these lines (the walls of the channel) holds by symmetry of the
initial state. In the pressure coordinate `η = p/p0`

    Φ = T0 g/Γ (1 - η^(RΓ/g)) + (u0 f0/2) [(y - y0) - Ly/2π sin(2πy/Ly)] ln η exp(-(ln η/b)²),
    u = -u0 sin²(πy/Ly) ln η exp(-(ln η/b)²),  T = -(η/R) ∂Φ/∂η,

with `T0 = 288 K`, `Γ = 0.005 K/m`, `u0 = 35 m/s`, `b = 2`, `y0 = Ly/2`; `η(z)`
by Newton in `ln η` from `Φ(η) = g z`. On the f-plane this is an exact steady state in
geostrophic and hydrostatic balance; the perturbation `u' = exp(-r²/Lp²)` m/s,
`Lp = 600 km`, at `(2000 km, 2500 km)` starts the wave.
"""
function baroclinic_channel(x, y, z; Ly=6e6, f0=1.0313e-4, perturbation::Bool=true)
    T0, Γ, u0, b = 288.0, 0.005, 35.0, 2.0
    mirrored = y > Ly
    ym = mirrored ? 2Ly - y : y
    A = u0 * f0 / 2 * ((ym - Ly / 2) - Ly / 2π * sin(2π * ym / Ly))
    temperature(η) = (l = log(η); T0 * η^(Rd * Γ / GRAV) - A / Rd * exp(-(l / b)^2) * (1 - 2l^2 / b^2))
    geopotential(η) = (l = log(η); T0 * GRAV / Γ * (1 - η^(Rd * Γ / GRAV)) + A * l * exp(-(l / b)^2))
    l = -GRAV * z / (Rd * T0)        # Newton in ln η: ∂Φ/∂ln η = -R T
    for _ in 1:50
        δ = (geopotential(exp(l)) - GRAV * z) / (-Rd * temperature(exp(l)))
        l -= δ
        abs(δ) < 1e-14 && break
    end
    η = exp(l)
    u = -u0 * sin(π * ym / Ly)^2 * l * exp(-(l / b)^2) * (mirrored ? -1 : 1)
    perturbation && (u += exp(-((x - 2e6)^2 + (y - 2.5e6)^2) / 6e5^2))
    return temperature(η), P0 * η, u, 0.0
end

function baroclinic_perturbation(lon, lat, z)
    d0, Vp, λc, φc, zp = 1 / 6, 1.0, π / 9, 2π / 9, 15000.0
    d = acos(clamp(sin(φc) * sin(lat) + cos(φc) * cos(lat) * cos(lon - λc), -1, 1))
    (z > zp || d > d0 || d == 0) && return 0.0, 0.0
    taper = 1 - 3z^2 / zp^2 + 2z^3 / zp^3
    s, c = sincos(π * d / (2d0))
    f = 16 / (3sqrt(3)) * Vp * taper * c^3 * s
    u = -f * (-sin(φc) * cos(lat) + cos(φc) * sin(lat) * cos(lon - λc)) / sin(d)
    v = f * cos(φc) * sin(lon - λc) / sin(d)
    return u, v
end

lonlat(x) = (atan(x[2], x[1]), asin(clamp(x[3] / norm(x), -1, 1)))

"Unit vectors east and north at the point `x` of the unit sphere."
function east_north(x)
    e = normalize(SVector(-x[2], x[1], 0.0))
    return e, cross(normalize(x), e)
end

"""
    balanced_column(profile, v, iap_theta) -> (ρ, ρθ)

Layer values of a column at rest in the discrete hydrostatic balance of the
vertical solver, `c_p θ_j (π_j - π_{j-1}) / dzh_j = -g`, with the temperature
of `profile(z) = (T, p, ...)` at the layer centres and its pressure at the
lowest one. `θ_j` at the interfaces is linearly interpolated, or with
`iap_theta` the upwind stencil at rest, which couples to the layers above
(fixed-point sweeps of the column).
"""
function balanced_column(profile, v::VerticalGrid, iap_theta::Bool)
    nlev, κ = num_layers(v), Rd / Cpd
    T = [profile(z)[1] for z in v.z_full]
    πk = [(profile(z)[2] / P0)^κ for z in v.z_full]
    for _ in 1:(iap_theta ? 30 : 1), j in 2:nlev
        c = iap_theta ? theta_stencil(j, 0.0, v.dz, v.dzh, v.wlow, nlev) : (0.0, v.wlow[j], 1 - v.wlow[j], 0.0)
        πj = πk[j]
        for _ in 1:50
            θh = sum(c[n] * T[j - 3 + n] / (j - 3 + n == j ? πj : πk[j - 3 + n]) for n in 1:4 if 1 <= j - 3 + n <= nlev)
            πj = πk[j - 1] - GRAV * v.dzh[j] / (Cpd * θh)
        end
        πk[j] = πj
    end
    ρθ = @. P0 / Rd * πk^(Cvd / Rd)
    return ρθ ./ (T ./ πk), ρθ
end

"""
    initial_state(m::CGridNonhydrostatic, profile)

State `(vn, w, ρ, ρθ)` from `profile(lon, lat, z) -> (T, p, u, v)`. Every
column is in discrete hydrostatic balance with the vertical operators of the
model: `π` of the lowest layer from `p`, above from
`c_p θ_j (π_j - π_{j-1}) / dzh_j = -g` with `θ = T/π` at the layer centres and
`θ_j` at the interfaces as the model has it at rest (fixed-point iteration per
level, sweeps over the column for the upwind `θ`), so a state without
horizontal motion stays at rest. `w = 0`.
"""
function initial_state(m::CGridNonhydrostatic, profile)
    g, v = m.grid, m.vgrid
    nc, ne, nlev = num_cells(g), num_edges(g), num_layers(v)
    ρ, ρθ = zeros(nc, nlev), zeros(nc, nlev)
    for i in 1:nc
        λ, φ = lonlat(g.centers[i])
        ρ[i, :], ρθ[i, :] = balanced_column(z -> profile(λ, φ, z), v, m.vertical_theta == 2)
    end
    vn = zeros(ne, nlev)
    for e in 1:ne
        x = g.edge_points[e]
        λ, φ = lonlat(x)
        east, north = east_north(x)
        n = g.edge_normals[e]
        for k in 1:nlev
            _, _, u, vv = profile(λ, φ, v.z_full[k])
            vn[e, k] = u * dot(east, n) + vv * dot(north, n)
        end
    end
    dev(x) = to_backend(m.backend, x)
    return (vn=dev(vn), w=KernelAbstractions.zeros(m.backend, Float64, nc, nlev + 1), ρ=dev(ρ), ρθ=dev(ρθ))
end
