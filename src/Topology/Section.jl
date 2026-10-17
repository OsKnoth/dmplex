"""
    FieldLayout(name, space, order, ncomp, ndofs)

Describes how many degrees of freedom a field places on each mesh point.
`ndofs(depth, cone_size)` is any callable returning the number of nodes on a
point of the given depth (0 vertex, 1 edge, 2 cell) whose cone has
`cone_size` entries; the finite-element families of the `FEM` module are such
callables. Layouts that depend on more than depth and cone size (e.g. on
point labels) extend [`num_nodes`](@ref) instead. Every node carries `ncomp` components. `space` is one of `:H1`,
`:L2`, `:Hdiv`, `:Hcurl` and decides how edge orientations act on the closure
(see [`closure_dofs`](@ref)).
"""
struct FieldLayout{F}
    name::String
    space::Symbol
    order::Int
    ncomp::Int
    ndofs::F
end

"""
    num_nodes(ndofs, dm, p)

Number of dof nodes a layout places on point `p`. Defaults to
`ndofs(depth, cone_size)`; extend it for point-dependent layouts.
"""
num_nodes(ndofs, dm::Plex, p::Integer) = ndofs(get_depth(dm, p), get_cone_size(dm, p))

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
    for p in chart, (f, field) in enumerate(fields)
        field_dof[f, p] = field.ncomp * num_nodes(field.ndofs, dm, p)
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
