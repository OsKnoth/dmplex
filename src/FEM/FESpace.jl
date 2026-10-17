"""
    FESpace(dm, family; name="u", ncomp=1, geometry=:sphere)
    FESpace(dm, section, field; geometry=:sphere)

Finite-element space of `family` on `dm`. The global numbering comes from a
`Section`, either built for this space alone or shared with other fields of
a mixed system. The reference element is chosen per cell from its number of
edges, so triangle, quadrilateral and mixed meshes are handled alike.

`geometry` selects the cell map: `:sphere` (curved elements, radially
projected onto the sphere) or `:chordal` (flat bilinear elements through the
vertices); see [`CellMap`](@ref).

Operators that depend only on the space (mesh, element, reference-to-sphere
map), such as the factorized mass matrix, are created on first use and cached
in `operators`; see [`mass_solver`](@ref).
"""
struct FESpace{F<:ElementFamily}
    dm::Plex
    section::Section
    field::Int
    family::F
    ncomp::Int
    elements::Dict{Int, ReferenceElement}  # keyed by the cell's number of edges
    operators::Dict{Any, Any}              # lazily built operators, e.g. (:mass, degree)
    geometry::Symbol                       # :sphere or :chordal
end

function FESpace(dm::Plex, section::Section, f::Integer; geometry::Symbol=:sphere)
    geometry in (:sphere, :chordal) || throw(ArgumentError("geometry must be :sphere or :chordal"))
    layout = section.fields[f]
    family = layout.ndofs
    family isa ElementFamily || throw(ArgumentError("field $(layout.name) has no finite-element family"))
    cone_sizes = unique(get_cone_size(dm, c) for c in cell_range(dm))
    elements = Dict{Int, ReferenceElement}(n => ReferenceElement(family, reference_shape(n)) for n in cone_sizes)
    return FESpace(dm, section, Int(f), family, layout.ncomp, elements, Dict{Any, Any}(), geometry)
end

FESpace(dm::Plex, section::Section, name::AbstractString; kwargs...) =
    FESpace(dm, section, field_index(section, name); kwargs...)

FESpace(dm::Plex, family::ElementFamily; name::AbstractString="u", ncomp::Integer=1, kwargs...) =
    FESpace(dm, Section(dm, field(family; name, ncomp)), 1; kwargs...)

"Reference-to-surface map of `cell` with the geometry of `V`."
Geometry.CellMap(V::FESpace, cell::Integer) = CellMap(V.dm, cell; projected=V.geometry == :sphere)

function_space(V::FESpace) = function_space(V.family)
num_dofs(V::FESpace) = storage_size(V.section)
reference_element(V::FESpace, cell::Integer) = V.elements[get_cone_size(V.dm, cell)]

"Global indices and orientation signs of the dofs of `cell` (see `closure_dofs`)."
cell_dofs(V::FESpace, cell::Integer) = closure_dofs(V.dm, V.section, cell, V.field)

"""
    evaluate(V, u, cell, ξ) -> (x, value)

Physical point and value of the discrete function with global coefficients
`u` at reference point `ξ` of `cell`. Scalar spaces return a number (or a
vector of `ncomp` numbers); H(div) and H(curl) spaces return the Cartesian
vector obtained with the contravariant/covariant Piola map.
"""
function evaluate(V::FESpace, u::AbstractVector, cell::Integer, ξ::SVector{2})
    el = reference_element(V, cell)
    indices, signs = cell_dofs(V, cell)
    values, _ = tabulate(el, [ξ])
    x, J, detJ = map_point(CellMap(V, cell), ξ)
    space = function_space(V)
    if space in (:H1, :L2)
        nc = V.ncomp
        result = [sum(u[indices[(a - 1) * nc + c]] * values[a, 1, 1] for a in 1:num_dofs(el)) for c in 1:nc]
        return x, nc == 1 ? result[1] : result
    end
    v̂ = sum(signs[a] * u[indices[a]] * SVector(values[a, 1, 1], values[a, 2, 1]) for a in 1:num_dofs(el))
    return x, space == :Hdiv ? piola_hdiv(J, detJ, v̂) : piola_hcurl(J, v̂)
end
