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
    initial_state(m::TRiSKNonhydrostatic, profile)

State `(vn, w, ρ, ρθ)` from `profile(lon, lat, z) -> (T, p, u, v)`. Every
column is in discrete hydrostatic balance with the vertical operators of the
model: `π` of the lowest layer from `p`, above from
`c_p θ_j (π_j - π_{j-1}) / dzh_j = -g` with `θ = T/π` at the layer centres
(fixed-point iteration per level), so a state without horizontal motion stays
at rest. `w = 0`.
"""
function initial_state(m::TRiSKNonhydrostatic, profile)
    g, v = m.grid, m.vgrid
    nc, ne, nlev = num_cells(g), num_edges(g), num_layers(v)
    ρ, ρθ = zeros(nc, nlev), zeros(nc, nlev)
    κ = Rd / Cpd
    for i in 1:nc
        λ, φ = lonlat(g.centers[i])
        T = [profile(λ, φ, z)[1] for z in v.z_full]
        πk = zeros(nlev)
        πk[1] = (profile(λ, φ, v.z_full[1])[2] / P0)^κ
        for j in 2:nlev
            πj = (profile(λ, φ, v.z_full[j])[2] / P0)^κ
            for _ in 1:50
                θh = v.wlow[j] * T[j - 1] / πk[j - 1] + (1 - v.wlow[j]) * T[j] / πj
                πj = πk[j - 1] - GRAV * v.dzh[j] / (Cpd * θh)
            end
            πk[j] = πj
        end
        for k in 1:nlev
            ρθ[i, k] = P0 / Rd * πk[k]^(Cvd / Rd)
            ρ[i, k] = ρθ[i, k] / (T[k] / πk[k])
        end
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
