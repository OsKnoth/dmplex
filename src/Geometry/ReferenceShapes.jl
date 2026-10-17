"""
Reference cells, following CGDycore: the triangle with vertices `(-1,-1)`,
`(1,-1)`, `(-1,1)` and the square `[-1,1]²`. Vertices are counter-clockwise
and local edge `i` runs from vertex `i` to vertex `i + 1`, matching the
closure order of a `Plex` cell.
"""
abstract type ReferenceShape end
struct Triangle <: ReferenceShape end
struct Quadrilateral <: ReferenceShape end

reference_vertices(::Triangle) = (SVector(-1.0, -1.0), SVector(1.0, -1.0), SVector(-1.0, 1.0))
reference_vertices(::Quadrilateral) =
    (SVector(-1.0, -1.0), SVector(1.0, -1.0), SVector(1.0, 1.0), SVector(-1.0, 1.0))

reference_area(::Triangle) = 2.0
reference_area(::Quadrilateral) = 4.0

"Reference shape of a cell with `n` edges."
function reference_shape(n::Integer)
    n == 3 && return Triangle()
    n == 4 && return Quadrilateral()
    throw(ArgumentError("no reference shape for cells with $n edges"))
end

reference_shape(dm::Plex, cell::Integer) = reference_shape(get_cone_size(dm, cell))

"""
    edge_point(shape, i, t)

Point on local edge `i` at parameter `t ∈ [-1, 1]`, running from vertex `i`
(`t = -1`) to vertex `i + 1` (`t = 1`).
"""
function edge_point(shape::ReferenceShape, i::Integer, t::Real)
    vertices = reference_vertices(shape)
    a, b = vertices[i], vertices[mod1(i + 1, length(vertices))]
    return ((1 - t) * a + (1 + t) * b) / 2
end

"Derivative of `edge_point` with respect to `t` (half the edge vector)."
function edge_tangent(shape::ReferenceShape, i::Integer)
    vertices = reference_vertices(shape)
    return (vertices[mod1(i + 1, length(vertices))] - vertices[i]) / 2
end

"Outward normal of local edge `i`, scaled like `edge_tangent` (tangent rotated clockwise)."
function edge_normal(shape::ReferenceShape, i::Integer)
    t = edge_tangent(shape, i)
    return SVector(t[2], -t[1])
end
