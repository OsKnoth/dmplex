"""
    CellMap(dm, cell; radius=1, projected=true)

Map from a reference cell onto a surface cell: the linear (triangle) or
bilinear (quadrilateral) interpolant of the cell's vertices. With
`projected=true` (curved elements) it is radially projected onto the sphere;
straight reference edges then map to great-circle arcs and the cells tile the
sphere exactly. With `projected=false` (chordal elements) the flat bilinear
surface through the vertices is used; neighbouring cells still match along
their straight chord edges, but the total area misses the sphere by O(h²).
On a plane the map is always the flat one, with cells unwrapped across
periodic seams.
"""
struct CellMap{S<:ReferenceShape, N}
    shape::S
    vertices::NTuple{N, SVector{3, Float64}}
    radius::Float64
    projected::Bool
end

function CellMap(dm::Plex, cell::Integer; radius::Real=1.0, projected::Bool=true)
    shape = reference_shape(dm, cell)
    vertices = Tuple(cell_coordinates(dm, cell))
    return CellMap(shape, vertices, Float64(radius), projected && dm.surface == :sphere)
end

function shape_functions(::Triangle, ξ)
    values = SVector(-(ξ[1] + ξ[2]) / 2, (1 + ξ[1]) / 2, (1 + ξ[2]) / 2)
    dξ = SVector(-0.5, 0.5, 0.0)
    dη = SVector(-0.5, 0.0, 0.5)
    return values, dξ, dη
end

function shape_functions(::Quadrilateral, ξ)
    x, y = ξ
    values = SVector((1 - x) * (1 - y), (1 + x) * (1 - y), (1 + x) * (1 + y), (1 - x) * (1 + y)) / 4
    dξ = SVector(-(1 - y), (1 - y), (1 + y), -(1 + y)) / 4
    dη = SVector(-(1 - x), -(1 + x), (1 + x), (1 - x)) / 4
    return values, dξ, dη
end

"""
    map_point(cmap, ξ) -> (x, J, detJ)

Physical point `x` on the sphere, the 3×2 Jacobian `J = ∂x/∂ξ` (columns are
tangent to the sphere) and the area element `detJ = |J₁ × J₂|`.
"""
function map_point(cmap::CellMap, ξ::SVector{2})
    N, dξ, dη = shape_functions(cmap.shape, ξ)
    X = sum(N[i] * cmap.vertices[i] for i in eachindex(N))
    Xξ = sum(dξ[i] * cmap.vertices[i] for i in eachindex(N))
    Xη = sum(dη[i] * cmap.vertices[i] for i in eachindex(N))
    if !cmap.projected
        J = cmap.radius * hcat(Xξ, Xη)
        return cmap.radius * X, J, norm(cross(J[:, 1], J[:, 2]))
    end
    r = norm(X)
    x = X / r
    # derivative of X / |X|: project out the radial part
    Jξ = cmap.radius * (Xξ - x * dot(x, Xξ)) / r
    Jη = cmap.radius * (Xη - x * dot(x, Xη)) / r
    J = hcat(Jξ, Jη)
    return cmap.radius * x, J, norm(cross(Jξ, Jη))
end

"Unit normal of the surface element with Jacobian `J` (outward for counter-clockwise cells)."
surface_normal(J) = normalize(cross(J[:, 1], J[:, 2]))

"Contravariant Piola transform for H(div): `J v̂ / detJ`."
piola_hdiv(J, detJ, v̂) = J * v̂ / detJ

"Covariant Piola transform for H(curl): `J (JᵀJ)⁻¹ v̂`."
piola_hcurl(J, v̂) = J * ((J' * J) \ v̂)
