const KITE_PRIMAL_VERTEX = 0
const KITE_EDGE_MIDPOINT = 1
const KITE_CENTER = 2
const KITE_HALF_EDGE = 1      # half of a primal edge, interior to a dual cell
const KITE_INTERIOR_EDGE = 2  # edge midpoint to cell center, interior to a primal cell

"""
    create_kite_mesh(primal; centers=nothing)

Kite grid of a polygonal `primal` mesh: every primal cell with `n` vertices
is split into `n` quadrilaterals (kites) by joining its center with the edge
midpoints. Kite `i` of a cell with counter-clockwise vertices `w₁ … wₙ` is

    (wᵢ, midpoint of edge i, center, midpoint of edge i - 1),

so in every kite the local vertex 1 is the primal vertex and local vertex 3 is
the cell center; local edges 1 and 4 are halves of primal edges, local edges
2 and 3 join an edge midpoint to the center. Kites sharing a primal vertex
form a dual cell.

Vertices are numbered primal vertices, then edge midpoints, then centers.
`centers` (one point per primal cell) defaults to the normalized vertex mean;
pass e.g. the MPAS or ICON cell generators instead. Labels:

- `"kite_parent_cell"`: primal cell index of each kite
- `"kite_dual_cell"`: primal vertex index (dual cell) of each kite
- `"kite_vertex_type"`: `0` primal vertex, `1` edge midpoint, `2` center
- `"kite_edge_type"`: `1` half of a primal edge, `2` midpoint–center edge
"""
function create_kite_mesh(primal::Plex; centers=nothing)
    nv, ne = num_vertices(primal), num_edges(primal)
    centers = centers === nothing ?
        [normalize(sum(get_coordinates(primal, v) for v in get_cell_vertices(primal, c))) for c in cell_range(primal)] :
        normalize.(collect(SVector{3, Float64}, centers))
    midpoints = [normalize(sum(get_coordinates(primal, v) for v in get_cone(primal, e))) for e in edge_range(primal)]
    coordinates = [primal.coordinates; midpoints; centers]
    midpoint_id(e) = nv + edge_index(primal, e)
    center_id(c) = nv + ne + cell_index(primal, c)

    kites = Vector{Int}[]
    parent_cell, dual_cell = Int[], Int[]
    for c in cell_range(primal)
        corners = get_cell_vertices(primal, c)
        edges = get_cone(primal, c)
        n = length(edges)
        for i in 1:n
            push!(kites, [vertex_index(primal, corners[i]), midpoint_id(edges[i]), center_id(c),
                          midpoint_id(edges[mod1(i - 1, n)])])
            push!(parent_cell, cell_index(primal, c))
            push!(dual_cell, vertex_index(primal, corners[i]))
        end
    end
    dm = plex_from_polygons(kites, coordinates)

    for (k, c) in enumerate(cell_range(dm))
        get_cell_vertices(dm, c)[1] == first(vertex_range(dm)) + kites[k][1] - 1 ||
            error("kite $k lost its canonical vertex order")
        set_label!(dm, "kite_parent_cell", c, parent_cell[k])
        set_label!(dm, "kite_dual_cell", c, dual_cell[k])
    end
    vertex_type(i) = i <= nv ? KITE_PRIMAL_VERTEX : i <= nv + ne ? KITE_EDGE_MIDPOINT : KITE_CENTER
    for v in vertex_range(dm)
        set_label!(dm, "kite_vertex_type", v, vertex_type(vertex_index(dm, v)))
    end
    for e in edge_range(dm)
        types = [get_label_value(dm, "kite_vertex_type", v) for v in get_cone(dm, e)]
        set_label!(dm, "kite_edge_type", e, KITE_CENTER in types ? KITE_INTERIOR_EDGE : KITE_HALF_EDGE)
    end
    return dm
end
