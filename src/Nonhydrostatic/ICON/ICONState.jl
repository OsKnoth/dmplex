# The state of a Fortran ICON run (native output on the triangles, model levels) in ICONNonhydrostatic.

"Index `idx[n]` of our point nearest to `pts[n]` (unit vectors), by buckets of size `h`; the largest distance."
function match_points(pts, ours, h)
    key(p) = round.(Int, p ./ h)
    buckets = Dict{NTuple{3, Int}, Vector{Int}}()
    for (i, p) in enumerate(ours)
        push!(get!(buckets, Tuple(key(p)), Int[]), i)
    end
    idx, dmax = zeros(Int, length(pts)), 0.0
    for (n, p) in enumerate(pts)
        k, best, bd = key(p), 0, Inf
        for a in -1:1, b in -1:1, c in -1:1, i in get(buckets, (k[1] + a, k[2] + b, k[3] + c), Int[])
            d = norm(ours[i] - p)
            d < bd && ((best, bd) = (i, d))
        end
        idx[n], dmax = best, max(dmax, bd)
    end
    return idx, dmax
end

"""
    read_icon_state(m::ICONNonhydrostatic, filename, gridfile; step=1) -> (vn, w, ρ, ρθ, exner)

The state of time `step` of an ICON output file on the native grid with the
model-level fields `vn`, `w`, `rho`, `theta_v`, `exner` (`output_grid =
.TRUE.`) on our grid and levels: cells and edges matched by their position,
`vn` by the sign of ICON's edge normals (`zonal_normal_primal_edge`,
`meridional_normal_primal_edge` of `gridfile`), ICON's levels from the top
reversed. `m.vgrid` must be ICON's (`icon_vertical_grid`).
"""
function read_icon_state(m::ICONNonhydrostatic, filename::AbstractString, gridfile::AbstractString; step::Integer=1)
    g, nlev = m.grid, num_layers(m.vgrid)
    unit(lon, lat) = SVector(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
    NCDataset(filename) do ds
        zifc = ds["z_ifc"][1, :]
        maximum(abs, sort(Float64.(zifc)) .- m.vgrid.z_half) < 1e-3 ||
            throw(ArgumentError("the levels of $filename are not those of the model"))
        h = 0.3 * minimum(g.dual_lengths)
        ci, dc = match_points(unit.(ds["clon"][:], ds["clat"][:]), g.centers, h)
        ei, de = match_points(unit.(ds["elon"][:], ds["elat"][:]), g.edge_points, h)
        max(dc, de) < 1e-8 && allunique(ci) && allunique(ei) ||
            throw(ArgumentError("the cells or edges of $filename are not those of the model grid"))
        zn, mn = NCDataset(f -> (f["zonal_normal_primal_edge"][:], f["meridional_normal_primal_edge"][:]), gridfile)
        s = map(eachindex(ei)) do n
            lon, lat = ds["elon"][n], ds["elat"][n]
            east, north = SVector(-sin(lon), cos(lon), 0.0), SVector(-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
            sign(dot(g.edge_normals[ei[n]], zn[n] * east + mn[n] * north))
        end
        function field(name, idx, sgn=1)
            x = ds[name][:, :, step]
            y = zeros(size(x))
            y[idx, :] .= sgn .* x[:, end:-1:1]
            return to_backend(m.backend, y)
        end
        ρ = field("rho", ci)
        return (vn=field("vn", ei, s), w=field("w", ci), ρ=ρ, ρθ=ρ .* field("theta_v", ci), exner=field("exner", ci))
    end
end
