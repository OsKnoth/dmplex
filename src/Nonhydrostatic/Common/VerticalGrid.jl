"""
    VerticalGrid(z_half)
    VerticalGrid(nlev, z_top; stretch=1)

Height levels of an extruded mesh, or of one column over orography: interfaces `z_half[j]`,
`j = 1:nlev+1`, from the ground (`z_half[1]`, zero for flat ground) to the top; layer centres
`z_full[k] = (z_half[k] + z_half[k+1]) / 2`, layer thicknesses `dz[k]`, and the
distances `dzh[j] = z_full[j] - z_full[j-1]` of the layer centres around the
interior interfaces (`dzh[1] = dzh[nlev+1] = 0`). `wlow[j]` is the weight of
the layer below in the linear interpolation to interface `j`,
`φ_j = wlow[j] φ_{j-1} + (1 - wlow[j]) φ_j`.

With `stretch > 1` the layers grow geometrically from the ground, the top
layer `stretch` times as thick as the lowest.
"""
struct VerticalGrid
    z_half::Vector{Float64}
    z_full::Vector{Float64}
    dz::Vector{Float64}
    dzh::Vector{Float64}
    wlow::Vector{Float64}
end

function VerticalGrid(z_half::AbstractVector{<:Real})
    z = collect(Float64, z_half)
    nlev = length(z) - 1
    nlev >= 1 && all(>(0), diff(z)) || throw(ArgumentError("z_half must increase"))
    zf = (z[1:nlev] .+ z[2:end]) ./ 2
    dz = diff(z)
    dzh, wlow = zeros(nlev + 1), zeros(nlev + 1)
    for j in 2:nlev
        dzh[j] = zf[j] - zf[j - 1]
        wlow[j] = (zf[j] - z[j]) / dzh[j]
    end
    return VerticalGrid(z, zf, dz, dzh, wlow)
end

function VerticalGrid(nlev::Integer, z_top::Real; stretch::Real=1)
    r = Float64(stretch)^(1 / max(nlev - 1, 1))
    dz = [r^(k - 1) for k in 1:nlev]
    return VerticalGrid([0.0; cumsum(dz) .* (z_top / sum(dz))])
end

num_layers(v::VerticalGrid) = length(v.dz)

"""
    terrain_following(vgrid, h; coordinate=:sleve, flat_height=16000, decay_scale=4000,
                      decay_exponent=1.35) -> z_half (columns × nlev+1)

Terrain-following interfaces `z = ζ + h_i b(ζ)` over the reference levels `ζ` of `vgrid` (flat,
from 0) for the surface heights `h`. `coordinate = :sleve`: the smooth level vertical decay of
Schär et al. (2002) with one scale, `b = sinh((F/s)^n - (ζ/s)^n)/sinh((F/s)^n)` up to
`F = flat_height` (`s = decay_scale`, `n = decay_exponent`, the values of ICON's
`decay_scale_1`, `decay_exp`, `flat_height`) and flat above; `:basic`: `b = 1 - ζ/z_top`
(Gal-Chen and Somerville 1975).
"""
function terrain_following(vgrid::VerticalGrid, h::AbstractVector{<:Real}; coordinate::Symbol=:sleve,
                           flat_height::Real=16000.0, decay_scale::Real=4000.0, decay_exponent::Real=1.35)
    ζ = vgrid.z_half
    ζ[1] == 0 || throw(ArgumentError("the reference levels must start at 0"))
    F, s, n = min(Float64(flat_height), ζ[end]), Float64(decay_scale), Float64(decay_exponent)
    b = coordinate == :basic ? 1 .- ζ ./ ζ[end] :
        coordinate == :sleve ? [z < F ? sinh((F / s)^n - (z / s)^n) / sinh((F / s)^n) : 0.0 for z in ζ] :
        throw(ArgumentError("coordinate must be :sleve or :basic"))
    z = ζ' .+ h .* b'
    all(>(0), diff(z; dims=2)) || throw(ArgumentError("the levels cross over the orography"))
    return z
end

"""
    model_levels(vgrid, centers, surface_height, z_half; coordinate, flat_height, decay_scale,
                 decay_exponent) -> z_half (cells × nlev+1)

The interfaces of the cells of a model: `z_half` if given, else terrain-following over
`surface_height(lon, lat)` (or the heights of the cells), else the flat levels of `vgrid`.
"""
function model_levels(vgrid::VerticalGrid, centers, surface_height, z_half; coordinate::Symbol=:sleve,
                      flat_height::Real=16000.0, decay_scale::Real=4000.0, decay_exponent::Real=1.35)
    nc, nlev = length(centers), num_layers(vgrid)
    if z_half !== nothing
        size(z_half) == (nc, nlev + 1) || throw(ArgumentError("z_half must be cells × nlev+1"))
        return Float64.(z_half)
    elseif surface_height !== nothing
        hs = surface_height isa AbstractVector ? surface_height : [surface_height(lonlat(x)...) for x in centers]
        return terrain_following(vgrid, hs; coordinate, flat_height, decay_scale, decay_exponent)
    end
    return repeat(vgrid.z_half', nc)
end

"""
    reference_atmosphere(z) -> (T, π)

ICON's reference atmosphere (`h_scal_bg = 10 km`, `t0sl_bg = 288.15 K`, `del_t_bg = 75 K`,
`p0sl_bg = 101325 Pa`, ICON's constants): the deviation `π - π_ref(z)` has no horizontal gradient
on height surfaces from the reference, which keeps the pressure gradient over orography accurate.
"""
function reference_atmosphere(z)
    p = 101325.0 * exp(-ICON_GRAV / ICON_RD * 10000.0 / 213.15 * log((exp(z / 10000.0) * 213.15 + 75.0) / 288.15))
    return 213.15 + 75.0 * exp(-z / 10000.0), (p / P0)^(ICON_RD / ICON_CPD)
end

"Weights `(q₁, q₂, q₃)` of the quadratic extrapolation of the lowest three layers of column `i` to the ground (ICON `wgtfacq`)."
function ground_extrapolation(c, i)
    size(c.z_full, 2) >= 3 || return (1.0, 0.0, 0.0)
    z1, z2, z3 = (c.z_full[i, k] - c.z_half[i, 1] for k in 1:3)
    q3 = z1 * z2 / (z2 - z3) / (z1 - z3)
    q2 = (z1 - q3 * (z1 - z3)) / (z1 - z2)
    return (1 - q2 - q3, q2, q3)
end

"""
    column_metrics(z_half) -> (; z_half, z_full, dz, dzh, wlow)

The metrics of `VerticalGrid` for every column of the interfaces `z_half` (columns × nlev+1), as
matrices (columns × layers or interfaces).
"""
function column_metrics(z_half::AbstractMatrix{<:Real})
    cols = [VerticalGrid(view(z_half, i, :)) for i in axes(z_half, 1)]
    stack(f) = permutedims(reduce(hcat, [getfield(c, f) for c in cols]))
    return (z_half=stack(:z_half), z_full=stack(:z_full), dz=stack(:dz), dzh=stack(:dzh), wlow=stack(:wlow))
end

"""
    icon_vertical_grid(nlev, top_height; min_lay_thckn=50, stretch_fac=1) -> VerticalGrid

ICON's height levels for flat ground (`init_sleve_coord`, `itype_laydistr = 1`
without the thickness limiter): `z = top_height (2/π arccos(((k-1)/nlev)^s))^x`
for the interfaces `k = 1:nlev+1` from the top, the exponent `x` such that the
lowest layer is `min_lay_thckn` thick.
"""
function icon_vertical_grid(nlev::Integer, top_height::Real; min_lay_thckn::Real=50.0, stretch_fac::Real=1.0)
    x = log(min_lay_thckn / top_height) / log(2 / π * acos(((nlev - 1) / nlev)^stretch_fac))
    z = [top_height * (2 / π * acos(((k - 1) / nlev)^stretch_fac))^x for k in 1:(nlev + 1)]
    z[end] = 0.0
    return VerticalGrid(reverse(z))
end
