# Shared helpers for the paper figures: evaluate kite fields densely and grid them.
using dmplex, LinearAlgebra, StaticArrays, NearestNeighbors, Plots, Printf, Serialization
const NH = dmplex.Nonhydrostatic
gr()
default(fontfamily="Computer Modern", framestyle=:box, grid=false, titlefontsize=11, guidefontsize=10,
        tickfontsize=8, legendfontsize=8, colorbar_titlefontsize=9)
const FIG = joinpath(@__DIR__, "..", "figures")              # PDF for the paper
const WEB = joinpath(@__DIR__, "..", "..", "images")          # PNG for the GitHub pages
mkpath(joinpath(@__DIR__, "data"))

"Save `p` as vector PDF (paper) and 300 dpi PNG (web)."
function save(p, name)
    savefig(p, joinpath(FIG, name * ".pdf"))
    savefig(plot(p; dpi=300), joinpath(WEB, name * ".png"))
end

"""
Samples `(x, v)` of the height-space field with the column values `vals` of the kite model `m`,
at n × n points of every kite (the Q2 interpolant, not the nodal values).
"""
function kite_samples(m, vals; n=8)
    H, d = m.horizontal.H, m.horizontal.distribution
    u = zeros(num_dofs(H))
    u[d.h.global_ids] .= vals
    t = [-1 + (2i - 1) / n for i in 1:n]
    xs, vs = SVector{3, Float64}[], Float64[]
    for c in cell_range(H.dm), a in t, b in t
        x, v = evaluate(H, u, c, SVector(a, b))
        push!(xs, x); push!(vs, v)
    end
    return xs, vs
end

"Inverse-distance weighting of the `k` nearest samples at the points `q`."
function idw(xs, vs, q; k=4)
    tree = KDTree(reduce(hcat, xs))
    idx, dist = knn(tree, reduce(hcat, q), k)
    return [let w = 1 ./ max.(dd, 1e-12) .^ 2; sum(w .* vs[ii]) / sum(w) end for (ii, dd) in zip(idx, dist)]
end

"Field on the lon-lat grid (degrees) from samples on the unit sphere."
function lonlat_grid(xs, vs, lon, lat)
    q = [SVector(cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ)) for φ in lat for λ in lon]
    return permutedims(reshape(idw(normalize.(xs), vs, q), length(lon), length(lat)))
end
