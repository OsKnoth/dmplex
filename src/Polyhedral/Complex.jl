# Cell complexes of polyhedra with vertex coordinates

"""
    PolyComplex

A 3D cell complex with vertex coordinates, the data the polyhedral finite volumes need: cells,
faces, edges and vertices numbered from 1 within each dimension; the cone of every cell (its
faces), face (its edges) and edge (its two vertices) with the incidence numbers `±1` of the
cellular boundary (`∂edge = v₂ - v₁`; the faces of a cell with `+1` are oriented outward). Cones
are stored in compressed form: the cone of cell `c` is `cell_faces[cell_offsets[c]:cell_offsets[c+1]-1]`.
Built by `extruded_complex` from an `ExtrudedPlex`; a general 3D plex fills the same arrays.
"""
struct PolyComplex
    cell_offsets::Vector{Int}
    cell_faces::Vector{Int}
    cell_signs::Vector{Int}
    face_offsets::Vector{Int}
    face_edges::Vector{Int}
    face_signs::Vector{Int}
    edge_vertices::Matrix{Int}        # 2 × edges, ∂e = v₂ - v₁
    coordinates::Vector{SVector{3, Float64}}
end

num_poly_cells(cx::PolyComplex) = length(cx.cell_offsets) - 1
num_poly_faces(cx::PolyComplex) = length(cx.face_offsets) - 1
num_poly_edges(cx::PolyComplex) = size(cx.edge_vertices, 2)
num_poly_vertices(cx::PolyComplex) = length(cx.coordinates)

cone_range(offsets, i) = offsets[i]:(offsets[i + 1] - 1)

"Faces of cell `c` and their signs (`+1`: oriented outward)."
cell_cone(cx::PolyComplex, c::Integer) = (view(cx.cell_faces, cone_range(cx.cell_offsets, c)),
                                          view(cx.cell_signs, cone_range(cx.cell_offsets, c)))

"Edges of face `f` and their signs in its boundary."
face_cone(cx::PolyComplex, f::Integer) = (view(cx.face_edges, cone_range(cx.face_offsets, f)),
                                          view(cx.face_signs, cone_range(cx.face_offsets, f)))

"""
    face_loop(cx, f) -> Vector{Int}

The vertices of face `f` in the cyclic order of its oriented boundary (the edges chained head to tail).
"""
function face_loop(cx::PolyComplex, f::Integer)
    edges, signs = face_cone(cx, f)
    oriented(i) = signs[i] > 0 ? (cx.edge_vertices[1, edges[i]], cx.edge_vertices[2, edges[i]]) :
                                 (cx.edge_vertices[2, edges[i]], cx.edge_vertices[1, edges[i]])
    n = length(edges)
    used = falses(n)
    a, b = oriented(1)
    loop, used[1] = [a], true
    while length(loop) < n
        i = findfirst(j -> !used[j] && oriented(j)[1] == b, 1:n)
        i === nothing && throw(ArgumentError("the boundary of face $f is not a closed loop"))
        used[i] = true
        push!(loop, b)
        b = oriented(i)[2]
    end
    b == a || throw(ArgumentError("the boundary of face $f is not a closed loop"))
    return loop
end

"""
    extruded_complex(ext::ExtrudedPlex, levels; radius=nothing) -> PolyComplex

The complex of the prisms of `ext` with the heights `levels` of its `num_levels(ext)` levels:
on the sphere the vertex `v` of the base at level `k` is at `(radius + levels[k]) v`, on a plane at
`v + levels[k] ẑ`. Cells, faces, edges and vertices are numbered in the order of the strata of `ext`.
"""
function extruded_complex(ext::ExtrudedPlex, levels::AbstractVector{<:Real}; radius::Union{Nothing, Real}=nothing)
    length(levels) == num_levels(ext) || throw(ArgumentError("levels needs $(num_levels(ext)) heights"))
    dm = ext.base
    sphere = dm.surface == :sphere
    sphere && radius === nothing && throw(ArgumentError("a spherical base needs a radius"))
    any(!iszero, dm.period) && throw(ArgumentError("periodic planes are not supported yet"))
    number(depth) = Dict(q => i for (i, q) in enumerate(get_depth_stratum(ext, depth)))
    cells, faces, edges, verts = number(3), number(2), number(1), number(0)
    function csr(points, index)
        offsets, cone, signs = [1], Int[], Int[]
        for q in points
            append!(cone, (index[r] for r in get_cone(ext, q)))
            append!(signs, get_cone_signs(ext, q))
            push!(offsets, length(cone) + 1)
        end
        return offsets, cone, signs
    end
    co, cf, cs = csr(get_depth_stratum(ext, 3), faces)
    fo, fe, fs = csr(get_depth_stratum(ext, 2), edges)
    ev = zeros(Int, 2, length(edges))
    for (i, q) in enumerate(get_depth_stratum(ext, 1))
        cone, signs = get_cone(ext, q), get_cone_signs(ext, q)
        a, b = signs[1] < 0 ? (cone[1], cone[2]) : (cone[2], cone[1])
        ev[:, i] .= (verts[a], verts[b])
    end
    coordinates = map(collect(get_depth_stratum(ext, 0))) do q
        p, k, _ = base_point(ext, q)
        x = get_coordinates(dm, p)
        sphere ? (radius + levels[k]) * x : x + SVector(0.0, 0.0, Float64(levels[k]))
    end
    return PolyComplex(co, cf, cs, fo, fe, fs, ev, coordinates)
end

"Vertices of cell `c` (each once)."
cell_vertices(cx::PolyComplex, c::Integer) = unique(reduce(vcat, (face_loop(cx, f) for f in cell_cone(cx, c)[1])))

"""
    face_cells(cx) -> Matrix{Int}

The two cells of every face, `2 × faces`: the cell out of which the face normal points (sign `+1`),
then the cell it points into (sign `-1`); `0` on the boundary.
"""
function face_cells(cx::PolyComplex)
    fc = zeros(Int, 2, num_poly_faces(cx))
    for c in 1:num_poly_cells(cx)
        faces, signs = cell_cone(cx, c)
        for (f, s) in zip(faces, signs)
            fc[s > 0 ? 1 : 2, f] = c
        end
    end
    return fc
end

"""
    vertex_neighbours(cx) -> Vector{Vector{Int}}

The cells that share a vertex with each cell (without the cell itself).
"""
function vertex_neighbours(cx::PolyComplex)
    nc = num_poly_cells(cx)
    verts = [cell_vertices(cx, c) for c in 1:nc]
    at = [Int[] for _ in 1:num_poly_vertices(cx)]
    for c in 1:nc, v in verts[c]
        push!(at[v], c)
    end
    return [setdiff(unique(reduce(vcat, (at[v] for v in verts[c]))), c) for c in 1:nc]
end
