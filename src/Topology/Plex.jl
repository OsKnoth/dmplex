"""
    Plex

Julia-native analogue of an interpolated PETSc `DMPlex` for surface meshes.

Points are numbered contiguously from 1 in height order, as PETSc does for
interpolated meshes: cells, then edges, then vertices. Cones and supports are
stored in compressed (CSR) arrays. Every cone entry carries a PETSc segment
orientation: `0` if the cell traverses the edge from its first to its second
vertex, `-1` if it traverses it in reverse.

Conventions:
- every edge is directed from its lower to its higher vertex point,
- every cell is oriented counter-clockwise seen from outside the sphere,
- hence the cell with orientation `0` on an edge lies to its *left*, and the
  edge normal `t × r` points from the left cell to the right cell.

`surface` is `:sphere` (unit sphere) or `:plane` (coordinates with `z = 0`).
A plane may be periodic with `period = (Lx, Ly)` on `[0, Lx) × [0, Ly)`
(`0` for a non-periodic direction); vertex coordinates are stored inside the
box and [`cell_coordinates`](@ref) unwraps cells that cross the seam.
"""
struct Plex
    strata::NTuple{3, UnitRange{Int}}  # indexed by depth + 1: vertices, edges, cells
    cone_offsets::Vector{Int}
    cones::Vector{Int}
    cone_orientations::Vector{Int}
    support_offsets::Vector{Int}
    supports::Vector{Int}
    coordinates::Vector{SVector{3, Float64}}
    labels::Dict{String, Dict{Int, Int}}
    surface::Symbol
    period::SVector{2, Float64}
end

# --- Construction -----------------------------------------------------------

"""
    plex_from_polygons(polygons, coordinates; period=nothing)

Build an interpolated `Plex` from a cell-vertex list (analogue of
`DMPlexCreateFromCellList` followed by `DMPlexInterpolate`). `polygons[c]`
lists the vertex indices (into `coordinates`) of cell `c` in cyclic order;
cells given clockwise are reversed so that all cells are counter-clockwise
seen from outside (from `+z` on a plane). Edges are numbered in order of first
appearance. Coordinates with `z = 0` define a plane, otherwise the unit
sphere; `period = (Lx, Ly)` makes a plane periodic.
"""
function plex_from_polygons(polygons::AbstractVector{<:AbstractVector{<:Integer}},
                            coordinates::AbstractVector{<:SVector{3, <:Real}}; period=nothing)
    vertex_coords = Vector{SVector{3, Float64}}(coordinates)
    surface = all(x -> x[3] == 0, vertex_coords) ? :plane : :sphere
    box = period === nothing ? SVector(0.0, 0.0) : SVector{2, Float64}(period)
    surface == :sphere && any(!iszero, box) && throw(ArgumentError("only planes can be periodic"))
    polys = [orient_ccw(collect(Int, poly), vertex_coords, surface, box) for poly in polygons]
    nc, nv = length(polys), length(vertex_coords)

    edge_ids = Dict{Tuple{Int, Int}, Int}()
    edge_vertices = Tuple{Int, Int}[]
    for poly in polys, i in eachindex(poly)
        key = minmax(poly[i], poly[mod1(i + 1, length(poly))])
        key[1] != key[2] || throw(ArgumentError("degenerate edge at vertex $(key[1])"))
        get!(edge_ids, key) do
            push!(edge_vertices, key)
            length(edge_vertices)
        end
    end
    ne = length(edge_vertices)
    edge_point(key) = nc + edge_ids[key]
    vertex_point(v) = nc + ne + v
    npoints = nc + ne + nv

    cone_offsets = Vector{Int}(undef, npoints + 1)
    cones = Int[]
    orientations = Int[]
    cone_offsets[1] = 1
    for (c, poly) in enumerate(polys)
        n = length(poly)
        for i in 1:n
            a, b = poly[i], poly[mod1(i + 1, n)]
            push!(cones, edge_point(minmax(a, b)))
            push!(orientations, a < b ? 0 : -1)
        end
        cone_offsets[c + 1] = length(cones) + 1
    end
    for (k, (a, b)) in enumerate(edge_vertices)
        push!(cones, vertex_point(a), vertex_point(b))
        push!(orientations, 0, 0)
        cone_offsets[nc + k + 1] = length(cones) + 1
    end
    cone_offsets[(nc + ne + 2):end] .= length(cones) + 1

    support_offsets, supports = invert_cones(cone_offsets, cones, npoints)
    strata = (nc + ne + 1:npoints, nc + 1:nc + ne, 1:nc)
    return Plex(strata, cone_offsets, cones, orientations, support_offsets, supports,
                vertex_coords, Dict{String, Dict{Int, Int}}(), surface, box)
end

function orient_ccw(poly::Vector{Int}, coordinates, surface::Symbol, period)
    x = [unwrap(period, coordinates[v], coordinates[poly[1]]) for v in poly]
    normal = sum(cross(x[i], x[mod1(i + 1, length(x))]) for i in eachindex(x))
    return dot(normal, reference_normal(surface, sum(x))) >= 0 ? poly : reverse(poly)
end

# --- Surface geometry -----------------------------------------------------------

"Shift `x` by multiples of `period` (where nonzero) to the copy closest to `reference`."
function unwrap(period, x::SVector{3}, reference::SVector{3})
    shift(k) = period[k] > 0 ? -period[k] * round((x[k] - reference[k]) / period[k]) : 0.0
    return x + SVector(shift(1), shift(2), 0.0)
end

"Outward unit normal of the surface at `x` (radial on the sphere, `+z` on a plane)."
reference_normal(surface::Symbol, x) = surface == :sphere ? normalize(x) : SVector(0.0, 0.0, 1.0)
reference_normal(dm, x) = reference_normal(dm.surface, x)

"""
    surface_point(dm, X)

Point of the surface represented by `X`: `X / |X|` on the sphere, `X` wrapped
into the periodic box on a plane.
"""
function surface_point(dm, X::SVector{3})
    dm.surface == :sphere && return normalize(X)
    wrap(k) = dm.period[k] > 0 ? mod(X[k], dm.period[k]) : X[k]
    return SVector(wrap(1), wrap(2), 0.0)
end

"Coordinates of the vertices of `cell` (counter-clockwise), unwrapped across periodic seams."
function cell_coordinates(dm, cell::Integer)
    vertices = get_cell_vertices(dm, cell)
    x1 = get_coordinates(dm, vertices[1])
    return [unwrap(dm.period, get_coordinates(dm, v), x1) for v in vertices]
end

"Coordinates of the two vertices of edge `e`, the second unwrapped next to the first."
function edge_coordinates(dm, e::Integer)
    a, b = (get_coordinates(dm, v) for v in get_cone(dm, e))
    return a, unwrap(dm.period, b, a)
end

function invert_cones(cone_offsets, cones, npoints)
    counts = zeros(Int, npoints)
    for q in cones
        counts[q] += 1
    end
    support_offsets = Vector{Int}(undef, npoints + 1)
    support_offsets[1] = 1
    for p in 1:npoints
        support_offsets[p + 1] = support_offsets[p] + counts[p]
    end
    supports = Vector{Int}(undef, length(cones))
    fill_pos = support_offsets[1:npoints]
    for p in 1:npoints, k in cone_offsets[p]:(cone_offsets[p + 1] - 1)
        q = cones[k]
        supports[fill_pos[q]] = p
        fill_pos[q] += 1
    end
    return support_offsets, supports
end

# --- Chart and strata ---------------------------------------------------------

get_chart(dm::Plex) = 1:(length(dm.cone_offsets) - 1)
get_depth_stratum(dm::Plex, depth::Integer) = dm.strata[depth + 1]
get_height_stratum(dm::Plex, height::Integer) = dm.strata[3 - height]

function get_depth(dm::Plex, p::Integer)
    for depth in 0:2
        p in dm.strata[depth + 1] && return depth
    end
    throw(BoundsError(get_chart(dm), p))
end

cell_range(dm::Plex) = dm.strata[3]
edge_range(dm::Plex) = dm.strata[2]
vertex_range(dm::Plex) = dm.strata[1]
num_cells(dm::Plex) = length(cell_range(dm))
num_edges(dm::Plex) = length(edge_range(dm))
num_vertices(dm::Plex) = length(vertex_range(dm))

"Position of a point inside its stratum, for indexing per-stratum arrays."
cell_index(dm::Plex, c::Integer) = c - first(cell_range(dm)) + 1
edge_index(dm::Plex, e::Integer) = e - first(edge_range(dm)) + 1
vertex_index(dm::Plex, v::Integer) = v - first(vertex_range(dm)) + 1

get_coordinates(dm::Plex, v::Integer) = dm.coordinates[vertex_index(dm, v)]

# --- Cones and supports -------------------------------------------------------

cone_span(dm::Plex, p::Integer) = dm.cone_offsets[p]:(dm.cone_offsets[p + 1] - 1)
get_cone(dm::Plex, p::Integer) = view(dm.cones, cone_span(dm, p))
get_cone_orientation(dm::Plex, p::Integer) = view(dm.cone_orientations, cone_span(dm, p))
get_cone_size(dm::Plex, p::Integer) = dm.cone_offsets[p + 1] - dm.cone_offsets[p]

get_support(dm::Plex, p::Integer) =
    view(dm.supports, dm.support_offsets[p]:(dm.support_offsets[p + 1] - 1))
get_support_size(dm::Plex, p::Integer) = dm.support_offsets[p + 1] - dm.support_offsets[p]

"`+1` if cone entry `i` of `cell` is traversed along the edge direction, else `-1`."
edge_sign(dm::Plex, cell::Integer, i::Integer) =
    dm.cone_orientations[dm.cone_offsets[cell] + i - 1] == 0 ? 1 : -1

"""
    get_edge_cells(dm, e) -> (left, right)

Cells to the left (orientation `0`) and right (orientation `-1`) of edge `e`;
a missing cell is reported as `0`.
"""
function get_edge_cells(dm::Plex, e::Integer)
    left = right = 0
    for c in get_support(dm, e)
        span = cone_span(dm, c)
        k = span[findfirst(==(e), view(dm.cones, span))]
        dm.cone_orientations[k] == 0 ? (left = c) : (right = c)
    end
    return left, right
end

"""
    get_cell_vertices(dm, cell)

Vertex points of `cell` in counter-clockwise order; vertex `i` is the start
of cone edge `i` in the cell's traversal direction.
"""
function get_cell_vertices(dm::Plex, cell::Integer)
    cone = get_cone(dm, cell)
    orientation = get_cone_orientation(dm, cell)
    return [get_cone(dm, cone[i])[orientation[i] == 0 ? 1 : 2] for i in eachindex(cone)]
end

"""
    get_closure(dm, p) -> (points, orientations)

Transitive closure in PETSc order: the point itself, its cone, then the cone's
vertices. For a cell the vertices follow `get_cell_vertices`.
"""
function get_closure(dm::Plex, p::Integer)
    depth = get_depth(dm, p)
    if depth == 0
        return [Int(p)], [0]
    elseif depth == 1
        return [Int(p); get_cone(dm, p)], [0, 0, 0]
    end
    cone = get_cone(dm, p)
    vertices = get_cell_vertices(dm, p)
    return [Int(p); cone; vertices], [0; get_cone_orientation(dm, p); zeros(Int, length(vertices))]
end

"Transitive support (star) of `p`, starting with `p` itself."
function get_star(dm::Plex, p::Integer)
    star = [Int(p)]
    frontier = [Int(p)]
    while !isempty(frontier)
        next = Int[]
        for q in frontier, s in get_support(dm, q)
            s in star || (push!(star, s); push!(next, s))
        end
        frontier = next
    end
    return star
end

"""
    get_adjacency(dm, p; use_closure=true)

Points adjacent to `p`, including `p`, sorted. With `use_closure=true` this is
the FEM adjacency of PETSc (star of the closure); with `use_closure=false` the
FVM adjacency (support of the cone).
"""
function get_adjacency(dm::Plex, p::Integer; use_closure::Bool=true)
    adjacent = Set{Int}((p,))
    if use_closure
        for q in first(get_closure(dm, p))
            union!(adjacent, get_star(dm, q))
        end
    else
        for q in get_cone(dm, p)
            union!(adjacent, get_support(dm, q))
        end
        union!(adjacent, get_support(dm, p))
    end
    return sort!(collect(adjacent))
end

"Cells sharing an edge with `cell`, sorted."
function get_cell_neighbors(dm::Plex, cell::Integer)
    neighbors = Int[]
    for e in get_cone(dm, cell), c in get_support(dm, e)
        c == cell || c in neighbors || push!(neighbors, c)
    end
    return sort!(neighbors)
end

"""
    boundary_matrix(dm, depth)

Signed incidence matrix of the boundary operator from points of `depth` to
points of `depth - 1`, with rows and columns in stratum order. For depth 1
the edge goes from its first vertex (`-1`) to its second (`+1`); for depth 2
the entry is `+1` for orientation `0` and `-1` otherwise. On a closed mesh
`boundary_matrix(dm, 1) * boundary_matrix(dm, 2) == 0`.
"""
function boundary_matrix(dm::Plex, depth::Integer)
    depth in (1, 2) || throw(ArgumentError("depth must be 1 or 2"))
    rows, cols, vals = Int[], Int[], Int[]
    lower = get_depth_stratum(dm, depth - 1)
    for (j, p) in enumerate(get_depth_stratum(dm, depth))
        cone = get_cone(dm, p)
        orientation = get_cone_orientation(dm, p)
        for i in eachindex(cone)
            push!(rows, cone[i] - first(lower) + 1)
            push!(cols, j)
            push!(vals, depth == 1 ? (i == 1 ? -1 : 1) : (orientation[i] == 0 ? 1 : -1))
        end
    end
    return sparse(rows, cols, vals, length(lower), length(get_depth_stratum(dm, depth)))
end

"""
    check_plex(dm; closed=true)

Verify the mesh invariants (cf. `DMPlexCheckSymmetry`/`DMPlexCheckSkeleton`):
cones and supports are symmetric, every cell boundary is a closed oriented
loop, cells are counter-clockwise, each edge has at most one cell per side,
and for `closed=true` the Euler characteristic is that of the sphere (or of
the torus for a doubly periodic plane).
"""
function check_plex(dm::Plex; closed::Bool=true)
    for p in get_chart(dm), q in get_cone(dm, p)
        p in get_support(dm, q) || error("cone/support mismatch between $p and $q")
    end
    for cell in cell_range(dm)
        vertices = get_cell_vertices(dm, cell)
        cone = get_cone(dm, cell)
        orientation = get_cone_orientation(dm, cell)
        length(cone) >= 3 || error("cell $cell has fewer than three edges")
        for i in eachindex(cone)
            edge = get_cone(dm, cone[i])
            stop = edge[orientation[i] == 0 ? 2 : 1]
            stop == vertices[mod1(i + 1, length(cone))] || error("cell $cell boundary is not a closed loop")
        end
        x = cell_coordinates(dm, cell)
        normal = sum(cross(x[i], x[mod1(i + 1, length(x))]) for i in eachindex(x))
        dot(normal, reference_normal(dm, sum(x))) > 0 || error("cell $cell is not counter-clockwise")
    end
    for e in edge_range(dm)
        nleft = count(c -> edge_sign(dm, c, findfirst(==(e), get_cone(dm, c))) == 1, get_support(dm, e))
        nright = get_support_size(dm, e) - nleft
        (nleft <= 1 && nright <= 1) || error("edge $e has inconsistent cell orientations")
        closed && (nleft, nright) != (1, 1) && error("edge $e is a boundary edge of a closed mesh")
    end
    if closed
        chi = num_vertices(dm) - num_edges(dm) + num_cells(dm)
        expected = dm.surface == :sphere ? 2 : 0  # sphere or doubly periodic plane (torus)
        chi == expected || error("Euler characteristic $chi != $expected")
    end
    return true
end

# --- Labels (DMLabel) ---------------------------------------------------------

function set_label!(dm::Plex, name::AbstractString, point::Integer, value::Integer)
    label = get!(dm.labels, String(name), Dict{Int, Int}())
    label[point] = value
    return dm
end

get_label_value(dm::Plex, name::AbstractString, point::Integer) =
    get(get(dm.labels, String(name), Dict{Int, Int}()), point, nothing)

function get_label_stratum(dm::Plex, name::AbstractString, value::Integer)
    label = get(dm.labels, String(name), Dict{Int, Int}())
    return sort!([point for (point, point_value) in label if point_value == value])
end

get_label_names(dm::Plex) = sort!(collect(keys(dm.labels)))

function clear_label!(dm::Plex, name::AbstractString)
    delete!(dm.labels, String(name))
    return dm
end
