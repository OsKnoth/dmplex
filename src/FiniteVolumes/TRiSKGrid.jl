"Great-circle distance of two points on the unit sphere."
arc_length(a, b) = atan(norm(cross(a, b)), dot(a, b))

"""
    TRiSKGrid(dm; centers=cell_centers(dm))

Geometry and connectivity of a C-grid finite-volume discretization on a
spherical mesh with an orthogonal dual (Voronoi–Delaunay pair: the hexagon
mesh, MPAS grids, the dual of ICON grids), as in ICON (hexagonal version) and
MPAS. Cells are the primal (Voronoi) polygons with mass points at the
generators `centers`; the dual cells are the triangles of three generators
around every mesh vertex, where vorticity lives; the normal velocity lives on
the edges, at the crossing of the primal edge and the dual edge.

Per edge `e` (cells `left → right`, normal `n_e` pointing from left to right):
primal length `l_e` (between its vertices), dual length `d_e` (between the
generators), the distances `d_{e,i}` of the two generators to the edge point.
Kites `K_{i,v}` (cell `i` ∩ dual cell of vertex `v`) are pairs of spherical
triangles (generator, edge point, vertex); cell areas `A_i` and dual areas
`A_v` are their sums, so the kite fractions `R_{i,v} = K_{i,v}/A_i` add up to
one in every cell (the TRiSK requirement).

All tables are padded matrices with one column per cell, vertex or edge;
unused entries have index `1` and weight `0`. `signs` of a cell are
`n_{e,i} = +1` if `n_e` points out of the cell; vertex `signs` are
`t_{e,v} = +1` if `n_e` is counter-clockwise around the vertex.
`trisk_weights[k, e]` are the weights `w_{e,e'}` of Thuburn et al. (2009) of
the edges `trisk_edges[k, e]` of the two cells of `e`: the tangential
velocity `(1/d_e) Σ w_{e,e'} l_{e'} u_{e'}` approximates `u ⋅ (r̂ × n_e)`, and
`w_{e,e'} = -w_{e',e}`.
"""
struct TRiSKGrid
    dm::Plex
    centers::Vector{SVector{3, Float64}}        # generators, by cell index
    vertices::Vector{SVector{3, Float64}}       # by vertex index
    edge_points::Vector{SVector{3, Float64}}
    edge_normals::Vector{SVector{3, Float64}}
    primal_lengths::Vector{Float64}
    dual_lengths::Vector{Float64}
    edge_cells::Matrix{Int}                     # (left, right) cell indices
    edge_vertices::Matrix{Int}
    cell_distances::Matrix{Float64}             # d_{e,left}, d_{e,right}
    cell_areas::Vector{Float64}
    vertex_areas::Vector{Float64}
    cell_edges::Matrix{Int}                     # counter-clockwise
    cell_signs::Matrix{Float64}
    cell_vertices::Matrix{Int}                  # vertex k: between cell edges k - 1 and k
    cell_kites::Matrix{Float64}
    vertex_edges::Matrix{Int}
    vertex_signs::Matrix{Float64}
    vertex_cells::Matrix{Int}
    vertex_kites::Matrix{Float64}
    trisk_edges::Matrix{Int}
    trisk_weights::Matrix{Float64}
end

num_cells(g::TRiSKGrid) = length(g.centers)
num_edges(g::TRiSKGrid) = length(g.edge_points)
num_vertices(g::TRiSKGrid) = length(g.vertices)

"Padded matrix of the vectors `columns` (`fill` in the unused entries)."
function padded(columns::Vector{<:Vector}, fill)
    T = promote_type(eltype(first(columns)), typeof(fill))
    A = Base.fill(T(fill), maximum(length, columns; init=0), length(columns))
    for (j, c) in enumerate(columns)
        A[eachindex(c), j] .= c
    end
    return A
end

function TRiSKGrid(dm::Plex; centers=nothing, tolerance::Real=1e-8)
    dm.surface == :sphere || throw(ArgumentError("TRiSKGrid needs a spherical mesh"))
    g = centers === nothing ? cell_centers(dm) : collect(SVector{3, Float64}, centers)
    defect = orthogonality_defect(dm, g)
    defect < tolerance ||
        @warn "the mesh has no orthogonal dual (defect $defect): TRiSK operators lose consistency"
    x = [get_coordinates(dm, v) for v in vertex_range(dm)]
    nc, ne, nv = num_cells(dm), num_edges(dm), num_vertices(dm)

    # edges: cells, vertices, the crossing of primal and dual edge, lengths
    edge_cells, edge_vertices = zeros(Int, 2, ne), zeros(Int, 2, ne)
    points, normals = Vector{SVector{3, Float64}}(undef, ne), Vector{SVector{3, Float64}}(undef, ne)
    l, d, distances = zeros(ne), zeros(ne), zeros(2, ne)
    for e in edge_range(dm)
        j = edge_index(dm, e)
        left, right = get_edge_cells(dm, e)
        (left == 0 || right == 0) && throw(ArgumentError("TRiSKGrid needs a closed mesh"))
        cl, cr = cell_index(dm, left), cell_index(dm, right)
        a, b = (vertex_index(dm, v) for v in get_cone(dm, e))
        edge_cells[:, j] .= (cl, cr)
        edge_vertices[:, j] .= (a, b)
        m = cross(cross(x[a], x[b]), cross(g[cl], g[cr]))
        m = norm(m) > 1e-14 ? normalize(m) : normalize(x[a] + x[b])
        dot(m, x[a] + x[b]) < 0 && (m = -m)
        points[j] = m
        normals[j] = normalize(cross(x[b], x[a]))     # from left to right (Plex convention)
        l[j], d[j] = arc_length(x[a], x[b]), arc_length(g[cl], g[cr])
        distances[:, j] .= (arc_length(g[cl], m), arc_length(g[cr], m))
    end

    # cells: edges counter-clockwise with outward signs, vertices with their kites
    cell_edges = [Int[] for _ in 1:nc]
    cell_signs = [Float64[] for _ in 1:nc]
    cell_vertices = [Int[] for _ in 1:nc]
    cell_kites = [Float64[] for _ in 1:nc]
    vertex_cells = [Int[] for _ in 1:nv]
    vertex_kites = [Float64[] for _ in 1:nv]
    for c in cell_range(dm)
        i = cell_index(dm, c)
        edges = [edge_index(dm, e) for e in get_cone(dm, c)]
        corners = [vertex_index(dm, v) for v in get_cell_vertices(dm, c)]  # corner k starts edge k
        n = length(edges)
        cell_edges[i] = edges
        cell_signs[i] = [Float64(edge_sign(dm, c, k)) for k in 1:n]
        cell_vertices[i] = corners
        cell_kites[i] = map(1:n) do k
            v = corners[k]
            spherical_triangle_area(g[i], points[edges[mod1(k - 1, n)]], x[v]) +
                spherical_triangle_area(g[i], points[edges[k]], x[v])
        end
        for (v, kite) in zip(corners, cell_kites[i])
            push!(vertex_cells[v], i)
            push!(vertex_kites[v], kite)
        end
    end
    cell_areas = sum.(cell_kites)
    vertex_areas = sum.(vertex_kites)

    # vertices: edges with their circulation signs
    vertex_edges = [[edge_index(dm, e) for e in get_support(dm, v)] for v in vertex_range(dm)]
    vertex_signs = [[sign(dot(normals[e], cross(x[v], points[e]))) for e in vertex_edges[v]] for v in 1:nv]

    # TRiSK weights: walk counter-clockwise around both cells of e, summing the kite fractions
    trisk_edges = [Int[] for _ in 1:ne]
    trisk_weights = [Float64[] for _ in 1:ne]
    for i in 1:nc
        edges, signs, n = cell_edges[i], cell_signs[i], length(cell_edges[i])
        R = cell_kites[i] ./ cell_areas[i]
        for k in 1:n
            e, S = edges[k], 0.0
            for s in 1:(n - 1)
                k′ = mod1(k + s, n)
                S += R[k′]               # corner k′ lies between cell edges k′ - 1 and k′
                push!(trisk_edges[e], edges[k′])
                push!(trisk_weights[e], signs[k] * signs[k′] * (0.5 - S))
            end
        end
    end

    return TRiSKGrid(dm, g, x, points, normals, l, d, edge_cells, edge_vertices, distances, cell_areas,
                     vertex_areas, padded(cell_edges, 1), padded(cell_signs, 0.0), padded(cell_vertices, 1),
                     padded(cell_kites, 0.0), padded(vertex_edges, 1), padded(vertex_signs, 0.0),
                     padded(vertex_cells, 1), padded(vertex_kites, 0.0), padded(trisk_edges, 1),
                     padded(trisk_weights, 0.0))
end
