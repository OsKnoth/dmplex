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

"Weights of the 1D rule `(x, w)` at coordinate `t` (which must be one of its nodes)."
function rule_weight(x, w, t)
    i = argmin(abs.(x .- t))
    abs(x[i] - t) < 1e-12 || error("node $t is not a quadrature node")
    return w[i]
end

# tensor GLL nodes on quadrilaterals are a quadrature rule: lumped mass matrices
function nodal_weights(f::Union{Lagrange, Discontinuous}, ::Quadrilateral, nodes)
    f.order == 0 && return [4.0]
    x, w = gauss_lobatto(f.order + 1)
    return [rule_weight(x, w, ξ[1]) * rule_weight(x, w, ξ[2]) for ξ in nodes]
end
