# Continuous and discontinuous nodal elements. Nodes are GLL based, so edge
# nodes of triangles and quadrilaterals coincide and are symmetric, which lets
# `closure_dofs` handle reversed edges by reversing the node order.

prime_basis(f::Union{Lagrange, Discontinuous}, shape::ReferenceShape) =
    pack_primes([[legendre_product(i, j, f.order)] for (i, j) in scalar_indices(f.order, shape)], f.order, 1)

function dof_functionals(f::Lagrange, shape::ReferenceShape)
    k = f.order
    g, _ = gauss_lobatto(k + 1)
    nodes = lattice_nodes(k, shape; interior=true)
    vertices = reference_vertices(shape)
    for i in eachindex(vertices), m in 2:k
        push!(nodes, edge_point(shape, i, g[m]))
    end
    append!(nodes, vertices)
    return point_value.(nodes), nodes
end

function dof_functionals(f::Discontinuous, shape::ReferenceShape)
    nodes = lattice_nodes(f.order, shape; interior=false)
    return point_value.(nodes), nodes
end
