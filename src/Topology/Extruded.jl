"""
    ExtrudedPlex(base, nlayers)

Extrusion of a surface `Plex` into `nlayers` layers, the analogue of PETSc's
`DMPlexExtrude` with tensor cells (`DM_POLYTOPE_TRI_PRISM_TENSOR`, …). As in
Firedrake (Lange et al. 2016, Bercea et al. 2016) only the surface plex and
the number of layers are stored; cones and supports of the 3D complex are
computed from those of the base.

Every 3D point is a pair of a base point `p` and either a layer
`k ∈ 1:nlayers` (the open interval between levels `k` and `k + 1`) or a level
`k ∈ 1:nlayers+1`; level 1 is the ground. The products are

| base   | × layer                 | × level            |
|--------|-------------------------|--------------------|
| cell   | 3D cell (tensor prism)  | horizontal face    |
| edge   | lateral face (quad)     | horizontal edge    |
| vertex | vertical edge           | 3D vertex          |

Points are numbered from 1 in height order like a PETSc plex (3D cells,
faces, edges, vertices); inside each block the column of a base point is
contiguous (`(i - 1) n + k`), so the vertical neighbours of a point are
consecutive numbers (Bercea et al. 2016).

The cone of a tensor cell is `(bottom, top, lateral faces in the order of the
base cone)`, that of a lateral face `(bottom edge, top edge, vertical edge at
the first vertex, vertical edge at the second vertex)`, that of a vertical edge
`(lower vertex, upper vertex)`. `get_cone_signs` gives the incidence numbers
of the cellular boundary, `∂(p × I) = ∂p × I + (-1)^dim(p) p × ∂I` with
`∂I = top - bottom`: the bottom face of a cell has `-1`, the top face `+1`
(outward with the base cell counter-clockwise from outside).
"""
struct ExtrudedPlex
    base::Plex
    nlayers::Int
    offsets::NTuple{6, Int}   # first point number - 1 of the blocks (cell, edge, vertex) × (layer, level)
    function ExtrudedPlex(base::Plex, nlayers::Integer)
        nlayers >= 1 || throw(ArgumentError("an extrusion needs at least one layer"))
        nc, ne, nv = num_cells(base), num_edges(base), num_vertices(base)
        n, m = nlayers, nlayers + 1
        # height order: 3D cells; faces (cell × level, edge × layer); edges (edge × level, vertex × layer); vertices
        cl = 0
        cv = cl + nc * n
        el = cv + nc * m
        ev = el + ne * n
        vl = ev + ne * m
        vv = vl + nv * n
        return new(base, n, (cl, el, vl, cv, ev, vv))
    end
end

num_layers(ext::ExtrudedPlex) = ext.nlayers
num_levels(ext::ExtrudedPlex) = ext.nlayers + 1

"Number of 3D points."
function get_chart(ext::ExtrudedPlex)
    nv = num_vertices(ext.base)
    return 1:(ext.offsets[6] + nv * num_levels(ext))
end

"""
Point range of block `b` of `ext.offsets`: (cell, edge, vertex) × layer for
`b = 1:3`, × level for `b = 4:6`.
"""
function block_range(ext::ExtrudedPlex, b::Integer)
    dm = ext.base
    count = (num_cells(dm), num_edges(dm), num_vertices(dm))[mod1(b, 3)]
    return (ext.offsets[b] + 1):(ext.offsets[b] + count * (b <= 3 ? ext.nlayers : ext.nlayers + 1))
end

base_depth_index(dm::Plex, p::Integer) = get_depth(dm, p) == 2 ? (1, cell_index(dm, p)) :
                                         get_depth(dm, p) == 1 ? (2, edge_index(dm, p)) : (3, vertex_index(dm, p))

"""
    extruded_point(ext, p, k, kind)

3D point of base point `p` and layer (`kind = :layer`) or level (`:level`) `k`.
"""
function extruded_point(ext::ExtrudedPlex, p::Integer, k::Integer, kind::Symbol)
    b, i = base_depth_index(ext.base, p)
    if kind == :layer
        1 <= k <= ext.nlayers || throw(BoundsError(1:ext.nlayers, k))
        return ext.offsets[b] + (i - 1) * ext.nlayers + k
    elseif kind == :level
        1 <= k <= ext.nlayers + 1 || throw(BoundsError(1:(ext.nlayers + 1), k))
        return ext.offsets[b + 3] + (i - 1) * (ext.nlayers + 1) + k
    end
    throw(ArgumentError("kind must be :layer or :level"))
end

"""
    base_point(ext, q) -> (p, k, kind)

Base point, layer or level index and `:layer`/`:level` of the 3D point `q`.
"""
function base_point(ext::ExtrudedPlex, q::Integer)
    q in get_chart(ext) || throw(BoundsError(get_chart(ext), q))
    dm = ext.base
    block = findfirst(b -> q in block_range(ext, b), 1:6)
    kind = block <= 3 ? :layer : :level
    n = kind == :layer ? ext.nlayers : ext.nlayers + 1
    i, k = divrem(q - ext.offsets[block] - 1, n) .+ (1, 1)
    b = block <= 3 ? block : block - 3
    p = b == 1 ? first(cell_range(dm)) + i - 1 : b == 2 ? first(edge_range(dm)) + i - 1 : first(vertex_range(dm)) + i - 1
    return p, k, kind
end

"Topological dimension of the 3D point `q`."
function get_depth(ext::ExtrudedPlex, q::Integer)
    p, _, kind = base_point(ext, q)
    return get_depth(ext.base, p) + (kind == :layer)
end

"""
    get_depth_stratum(ext, depth)

3D points of the given depth (an iterator over two contiguous blocks for the
faces and edges).
"""
function get_depth_stratum(ext::ExtrudedPlex, depth::Integer)
    block(b) = block_range(ext, b)
    depth == 3 && return block(1)
    depth == 2 && return Iterators.flatten((block(4), block(2)))
    depth == 1 && return Iterators.flatten((block(5), block(3)))
    depth == 0 && return block(6)
    throw(ArgumentError("depth must be 0:3"))
end

"""
    get_cone(ext, q) -> Vector{Int}

Cone of the 3D point `q` (see [`ExtrudedPlex`](@ref) for the order).
"""
function get_cone(ext::ExtrudedPlex, q::Integer)
    dm = ext.base
    p, k, kind = base_point(ext, q)
    base_cone = get_cone(dm, p)
    if kind == :level
        return [extruded_point(ext, r, k, :level) for r in base_cone]
    end
    bottom, top = extruded_point(ext, p, k, :level), extruded_point(ext, p, k + 1, :level)
    return [bottom; top; [extruded_point(ext, r, k, :layer) for r in base_cone]]
end

"""
    get_cone_signs(ext, q) -> Vector{Int}

Incidence numbers `±1` of the cone of `q` in the cellular boundary.
"""
function get_cone_signs(ext::ExtrudedPlex, q::Integer)
    dm = ext.base
    p, _, kind = base_point(ext, q)
    d = get_depth(dm, p)
    base_signs = d == 2 ? [edge_sign(dm, p, i) for i in 1:get_cone_size(dm, p)] :
                 d == 1 ? [-1, 1] : Int[]
    kind == :level && return base_signs
    s = iseven(d) ? 1 : -1                 # (-1)^dim(p)
    return [-s; s; base_signs]
end

"""
    get_support(ext, q) -> Vector{Int}

3D points whose cone contains `q`.
"""
function get_support(ext::ExtrudedPlex, q::Integer)
    dm = ext.base
    p, k, kind = base_point(ext, q)
    lateral = [extruded_point(ext, r, k, kind) for r in get_support(dm, p)]
    kind == :layer && return lateral
    vertical = [extruded_point(ext, p, j, :layer) for j in (k - 1, k) if 1 <= j <= ext.nlayers]
    return [vertical; lateral]
end

"""
    boundary_matrix(ext, depth)

Sparse incidence matrix of the 3D complex from the points of `depth` to those
of `depth - 1` (rows and columns in stratum order); `boundary_matrix(ext, d) *
boundary_matrix(ext, d + 1) == 0`.
"""
function boundary_matrix(ext::ExtrudedPlex, depth::Integer)
    1 <= depth <= 3 || throw(ArgumentError("depth must be 1:3"))
    rows_of = Dict(q => i for (i, q) in enumerate(get_depth_stratum(ext, depth - 1)))
    I, J, V = Int[], Int[], Int[]
    for (j, q) in enumerate(get_depth_stratum(ext, depth))
        for (r, s) in zip(get_cone(ext, q), get_cone_signs(ext, q))
            push!(I, rows_of[r]); push!(J, j); push!(V, s)
        end
    end
    return sparse(I, J, V, length(rows_of), count(_ -> true, get_depth_stratum(ext, depth)))
end
