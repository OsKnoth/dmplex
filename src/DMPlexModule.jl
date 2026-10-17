module DMPlexModule

export MyDMPlex, add_cell_cone!, add_edge_cone!, get_cone, get_support_cells,
    get_support_points, get_transitive_closure, get_adjacency, get_cell_neighbors,
    set_cell_vertices, get_cell_vertices, set_label!, get_label_value,
    get_label_stratum, get_label_names, clear_label!, set_field!, get_field,
    get_field_names, remove_field!

abstract type AbstractPointField end

mutable struct PointField{T} <: AbstractPointField
    values::Dict{Int, Vector{T}}
end

mutable struct MyDMPlex
    cell_start::Int; cell_end::Int
    edge_start::Int; edge_end::Int
    vertex_start::Int; vertex_end::Int
    
    cones::Dict{Int, Vector{Int}}
    edge_to_cells::Dict{Int, Tuple{Int, Int}}
    cell_to_vertices::Dict{Int, Vector{Int}}
    supports::Dict{Int, Vector{Int}}
    labels::Dict{String, Dict{Int, Int}}
    fields::Dict{String, AbstractPointField}

    function MyDMPlex()
        new(0, 0, 0, 0, 0, 0, 
            Dict{Int, Vector{Int}}(), 
            Dict{Int, Tuple{Int, Int}}(),
            Dict{Int, Vector{Int}}(),
            Dict{Int, Vector{Int}}(),
            Dict{String, Dict{Int, Int}}(),
            Dict{String, AbstractPointField}())
    end
end

function add_edge_cone!(dm::MyDMPlex, edge_point::Int, vertex_points::Vector{Int})
    dm.cones[edge_point] = copy(vertex_points)
    for vertex_point in vertex_points
        support = get!(dm.supports, vertex_point, Int[])
        edge_point in support || push!(support, edge_point)
    end
end

function add_cell_cone!(dm::MyDMPlex, cell_point::Int, oriented_edges::Vector{Int})
    for oriented_edge in oriented_edges
        oriented_edge != 0 || throw(ArgumentError("oriented edge point cannot be zero"))
        edge_point = abs(oriented_edge)
        left, right = get(dm.edge_to_cells, edge_point, (0, 0))
        if oriented_edge > 0
            left == 0 || left == cell_point || throw(ArgumentError("edge $edge_point has more than one positively oriented support cell"))
        else
            right == 0 || right == cell_point || throw(ArgumentError("edge $edge_point has more than one negatively oriented support cell"))
        end
    end
    dm.cones[cell_point] = copy(oriented_edges)
    for oriented_edge in oriented_edges
        edge_point = abs(oriented_edge)
        left, right = get(dm.edge_to_cells, edge_point, (0, 0))
        if oriented_edge > 0
            dm.edge_to_cells[edge_point] = (cell_point, right)
        else
            dm.edge_to_cells[edge_point] = (left, cell_point)
        end
        support = get!(dm.supports, edge_point, Int[])
        cell_point in support || push!(support, cell_point)
    end
end

function get_cone(dm::MyDMPlex, point::Int)
    return get(dm.cones, point, Int[])
end

function get_support_cells(dm::MyDMPlex, edge_point::Int)
    return get(dm.edge_to_cells, edge_point, (0, 0))
end

function get_support_points(dm::MyDMPlex, point::Int)
    return get(dm.supports, point, Int[])
end

function get_transitive_closure(dm::MyDMPlex, point::Int)
    closure = Int[]
    visited = Set{Int}()
    stack = [point]
    while !isempty(stack)
        current = pop!(stack)
        current in visited && continue
        push!(visited, current)
        push!(closure, current)
        append!(stack, abs.(get_cone(dm, current)))
    end
    return closure
end

function get_adjacency(dm::MyDMPlex, point::Int)
    adjacent = Set{Int}()
    for closure_point in get_transitive_closure(dm, point)
        for support_point in get_support_points(dm, closure_point)
            support_point == point || push!(adjacent, support_point)
        end
    end
    delete!(adjacent, point)
    return sort!(collect(adjacent))
end

function get_cell_neighbors(dm::MyDMPlex, cell_point::Int)
    neighbors = Set{Int}()
    for oriented_edge in get_cone(dm, cell_point)
        for support_cell in get_support_points(dm, abs(oriented_edge))
            support_cell == cell_point || push!(neighbors, support_cell)
        end
    end
    return sort!(collect(neighbors))
end

function set_cell_vertices(dm::MyDMPlex, cell_point::Int, vertex_points::Vector{Int})
    dm.cell_to_vertices[cell_point] = copy(vertex_points)
end

function get_cell_vertices(dm::MyDMPlex, cell_point::Int)
    return get(dm.cell_to_vertices, cell_point, Int[])
end

function set_label!(dm::MyDMPlex, name::AbstractString, point::Int, value::Integer)
    label = get!(dm.labels, String(name), Dict{Int, Int}())
    label[point] = value
    return dm
end

function get_label_value(dm::MyDMPlex, name::AbstractString, point::Int)
    return get(get(dm.labels, String(name), Dict{Int, Int}()), point, nothing)
end

function get_label_stratum(dm::MyDMPlex, name::AbstractString, value::Integer)
    label = get(dm.labels, String(name), Dict{Int, Int}())
    return sort!([point for (point, point_value) in label if point_value == value])
end

function get_label_names(dm::MyDMPlex)
    return sort!(collect(keys(dm.labels)))
end

function clear_label!(dm::MyDMPlex, name::AbstractString)
    delete!(dm.labels, String(name))
    return dm
end

function set_field!(dm::MyDMPlex, name::AbstractString, point::Int, values::AbstractVector{T}) where {T}
    field_name = String(name)
    if haskey(dm.fields, field_name)
        field = dm.fields[field_name]
        field isa PointField{T} ||
            throw(ArgumentError("field $field_name already stores values of a different element type"))
        field.values[point] = collect(values)
    else
        dm.fields[field_name] = PointField(Dict(point => collect(values)))
    end
    return dm
end

set_field!(dm::MyDMPlex, name::AbstractString, point::Int, value::Number) =
    set_field!(dm, name, point, [value])

function get_field(dm::MyDMPlex, name::AbstractString, point::Int)
    field = dm.fields[String(name)]
    return copy(field.values[point])
end

function get_field_names(dm::MyDMPlex)
    return sort!(collect(keys(dm.fields)))
end

function remove_field!(dm::MyDMPlex, name::AbstractString)
    delete!(dm.fields, String(name))
    return dm
end

end # module
