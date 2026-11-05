"""
    VerticalGrid(z_half)
    VerticalGrid(nlev, z_top; stretch=1)

Height levels of an extruded mesh with flat ground: interfaces `z_half[j]`,
`j = 1:nlev+1`, from the ground (`z_half[1] = 0`) to the top; layer centres
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
    nlev >= 1 && issorted(z) && z[1] == 0 || throw(ArgumentError("z_half must increase from 0"))
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
