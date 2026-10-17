module SectionModule

using ..DMPlexModule

export FieldLayout, lagrange, discontinuous, raviart_thomas, nedelec,
    Section, num_fields, field_names, field_index, get_dof, get_offset,
    get_field_dof, get_field_offset, storage_size, dof_range, field_range,
    closure_dofs, create_vector

"""
    FieldLayout(name, space, order, ncomp, ndofs)

Describes how many degrees of freedom a finite-element field places on each
mesh point. `ndofs(depth, cone_size)` returns the number of nodes on a point of
the given depth (0 vertex, 1 edge, 2 cell) whose cone has `cone_size` entries;
every node carries `ncomp` components. `space` is one of `:H1`, `:L2`, `:Hdiv`,
`:Hcurl` and decides how edge orientations act on the closure (see
[`closure_dofs`](@ref)).
"""
struct FieldLayout{F}
    name::String
    space::Symbol
    order::Int
    ncomp::Int
    ndofs::F
end

const TRIANGLE = 3
const QUADRILATERAL = 4

unsupported(name, k, n) = throw(ArgumentError("$name of order $k is not defined on cells with $n edges"))

"Continuous Lagrange space of order `k ≥ 1` (P_k on triangles, Q_k on quadrilaterals)."
function lagrange(k::Integer; name::AbstractString="u", ncomp::Integer=1)
    k >= 1 || throw(ArgumentError("Lagrange order must be at least 1"))
    ndofs(depth, n) = depth == 0 ? 1 :
                      depth == 1 ? k - 1 :
                      k == 1 ? 0 :
                      n == TRIANGLE ? (k - 1) * (k - 2) ÷ 2 :
                      n == QUADRILATERAL ? (k - 1)^2 : unsupported("Lagrange", k, n)
    return FieldLayout(String(name), :H1, Int(k), Int(ncomp), ndofs)
end

"Discontinuous space of order `k ≥ 0`, all nodes owned by the cell (DG0 on any polygon)."
function discontinuous(k::Integer; name::AbstractString="h", ncomp::Integer=1)
    k >= 0 || throw(ArgumentError("DG order must be non-negative"))
    ndofs(depth, n) = depth < 2 ? 0 :
                      k == 0 ? 1 :
                      n == TRIANGLE ? (k + 1) * (k + 2) ÷ 2 :
                      n == QUADRILATERAL ? (k + 1)^2 : unsupported("DG", k, n)
    return FieldLayout(String(name), :L2, Int(k), Int(ncomp), ndofs)
end

function edge_element_dofs(name, k)
    ndofs(depth, n) = depth == 0 ? 0 :
                      depth == 1 ? k + 1 :
                      k == 0 ? 0 :
                      n == TRIANGLE ? k * (k + 1) :
                      n == QUADRILATERAL ? 2k * (k + 1) : unsupported(name, k, n)
    return ndofs
end

"""
Raviart–Thomas space `RT_k`, `k ≥ 0`, with divergence in P_k/Q_k (RTCF on
quadrilaterals). Edge nodes are normal components ordered along the edge.
"""
function raviart_thomas(k::Integer; name::AbstractString="u")
    k >= 0 || throw(ArgumentError("Raviart–Thomas order must be non-negative"))
    return FieldLayout(String(name), :Hdiv, Int(k), 1, edge_element_dofs("Raviart–Thomas", k))
end

"Nédélec space of the first kind `N_k`, `k ≥ 0`; edge nodes are tangential components."
function nedelec(k::Integer; name::AbstractString="u")
    k >= 0 || throw(ArgumentError("Nédélec order must be non-negative"))
    return FieldLayout(String(name), :Hcurl, Int(k), 1, edge_element_dofs("Nédélec", k))
end

"""
    Section(dm, fields...; ordering=:point)

Analogue of `PetscSection`: assigns each field a number of dofs per mesh point
and a 0-based offset into a global vector. `ordering=:point` stores all fields
of a point contiguously (PETSc default); `ordering=:field` stores each field
as one contiguous block, which suits block preconditioners for mixed systems.
"""
struct Section
    fields::Vector{FieldLayout}
    ordering::Symbol
    field_dof::Matrix{Int}     # (field, point)
    field_offset::Matrix{Int}  # (field, point), 0-based
    field_start::Vector{Int}   # 0-based start of each field block (:field ordering)
    size::Int
end

function Section(dm::Plex, fields::FieldLayout...; ordering::Symbol=:point)
    ordering in (:point, :field) || throw(ArgumentError("ordering must be :point or :field"))
    isempty(fields) && throw(ArgumentError("a section needs at least one field"))
    chart = get_chart(dm)
    nf = length(fields)
    field_dof = zeros(Int, nf, length(chart))
    for p in chart
        depth, n = get_depth(dm, p), get_cone_size(dm, p)
        for (f, field) in enumerate(fields)
            field_dof[f, p] = field.ncomp * field.ndofs(depth, n)
        end
    end
    field_offset = similar(field_dof)
    field_start = zeros(Int, nf)
    offset = 0
    if ordering == :point
        for p in chart, f in 1:nf
            field_offset[f, p] = offset
            offset += field_dof[f, p]
        end
    else
        for f in 1:nf
            field_start[f] = offset
            for p in chart
                field_offset[f, p] = offset
                offset += field_dof[f, p]
            end
        end
    end
    return Section(collect(FieldLayout, fields), ordering, field_dof, field_offset, field_start, offset)
end

num_fields(s::Section) = length(s.fields)
field_names(s::Section) = [field.name for field in s.fields]

function field_index(s::Section, name::AbstractString)
    f = findfirst(field -> field.name == name, s.fields)
    f === nothing && throw(KeyError(name))
    return f
end

storage_size(s::Section) = s.size
create_vector(s::Section, ::Type{T}=Float64) where {T} = zeros(T, storage_size(s))

get_dof(s::Section, p::Integer) = sum(view(s.field_dof, :, p))
get_field_dof(s::Section, p::Integer, f::Integer) = s.field_dof[f, p]
get_field_offset(s::Section, p::Integer, f::Integer) = s.field_offset[f, p]

"0-based offset of the dofs of point `p`; only defined for `:point` ordering."
function get_offset(s::Section, p::Integer)
    s.ordering == :point || throw(ArgumentError("point offsets require :point ordering"))
    return s.field_offset[1, p]
end

"1-based global indices of the dofs of point `p` (all fields, or field `f`)."
dof_range(s::Section, p::Integer) = (get_offset(s, p) + 1):(get_offset(s, p) + get_dof(s, p))
dof_range(s::Section, p::Integer, f::Integer) =
    (s.field_offset[f, p] + 1):(s.field_offset[f, p] + s.field_dof[f, p])

"1-based global index block of field `f`; only defined for `:field` ordering."
function field_range(s::Section, f::Integer)
    s.ordering == :field || throw(ArgumentError("field blocks require :field ordering"))
    stop = f < num_fields(s) ? s.field_start[f + 1] : s.size
    return (s.field_start[f] + 1):stop
end

"""
    closure_dofs(dm, s, cell, f) -> (indices, signs)

Global 1-based indices of field `f` on the closure of `cell`, in PETSc closure
order (cell interior, edges in cone order, vertices counter-clockwise). On
each point the nodes are node-major (`ncomp` components per node).

Edge nodes are stored in the edge's global direction. For an edge the cell
traverses in reverse (orientation `-1`) the node order is reversed so that the
local numbering always follows the cell's counter-clockwise traversal, and
for `:Hdiv`/`:Hcurl` fields the sign becomes `-1`, mapping the global edge
normal/tangent onto the cell's outward normal/counter-clockwise tangent.
Neighbouring cells thus see a shared edge with opposite node order and sign.
"""
function closure_dofs(dm::Plex, s::Section, cell::Integer, f::Integer)
    field = s.fields[f]
    flips_sign = field.space in (:Hdiv, :Hcurl)
    points, orientations = get_closure(dm, cell)
    indices = Int[]
    signs = Int[]
    for (p, orientation) in zip(points, orientations)
        ndof = s.field_dof[f, p]
        ndof == 0 && continue
        base = s.field_offset[f, p]
        nodes = ndof ÷ field.ncomp
        sign = flips_sign && orientation != 0 ? -1 : 1
        for node in (orientation == 0 ? (1:nodes) : (nodes:-1:1)), comp in 1:field.ncomp
            push!(indices, base + (node - 1) * field.ncomp + comp)
            push!(signs, sign)
        end
    end
    return indices, signs
end

closure_dofs(dm::Plex, s::Section, cell::Integer, name::AbstractString) =
    closure_dofs(dm, s, cell, field_index(s, name))

end # module
