# Raviart–Thomas (H(div)) and Nédélec (H(curl)) elements of arbitrary order.
#
# Edge dofs are the normal (RT) or tangential (Nédélec) component at the k + 1
# Gauss points of each edge, weighted with the scaled reference normal/tangent.
# Under the Piola maps these equal flux and circulation densities in physical
# space, so they are shared between neighbouring cells; reversing an edge
# reverses the (symmetric) points and flips the sign, as `closure_dofs` does.
# Interior dofs are moments against Legendre products.

zero_coeffs(D) = zeros(D + 1, D + 1)

function prime_basis(f::RaviartThomas, shape::Triangle)
    k, D = f.order, f.order + 1
    functions = Vector{Vector{Matrix{Float64}}}()
    for (i, j) in scalar_indices(k, shape)
        push!(functions, [legendre_product(i, j, D), zero_coeffs(D)], [zero_coeffs(D), legendre_product(i, j, D)])
    end
    for a in 0:k  # x ⋅ P̃_k completes [P_k]² to RT_k
        c = legendre_product(a, k - a, D)
        push!(functions, [mul_x(c), mul_y(c)])
    end
    return pack_primes(functions, D, 2)
end

function prime_basis(f::Nedelec, shape::Triangle)
    k, D = f.order, f.order + 1
    functions = Vector{Vector{Matrix{Float64}}}()
    for (i, j) in scalar_indices(k, shape)
        push!(functions, [legendre_product(i, j, D), zero_coeffs(D)], [zero_coeffs(D), legendre_product(i, j, D)])
    end
    for a in 0:k  # (-y, x) ⋅ P̃_k: the rotated Raviart–Thomas complement
        c = legendre_product(a, k - a, D)
        push!(functions, [-mul_y(c), mul_x(c)])
    end
    return pack_primes(functions, D, 2)
end

# RTCF_k = Q_{k+1,k} × Q_{k,k+1};  NCE_k = Q_{k,k+1} × Q_{k+1,k}
function tensor_vector_primes(k, first_degrees, second_degrees)
    D = k + 1
    functions = Vector{Vector{Matrix{Float64}}}()
    for j in 0:first_degrees[2], i in 0:first_degrees[1]
        push!(functions, [legendre_product(i, j, D), zero_coeffs(D)])
    end
    for j in 0:second_degrees[2], i in 0:second_degrees[1]
        push!(functions, [zero_coeffs(D), legendre_product(i, j, D)])
    end
    return pack_primes(functions, D, 2)
end

prime_basis(f::RaviartThomas, ::Quadrilateral) = tensor_vector_primes(f.order, (f.order + 1, f.order), (f.order, f.order + 1))
prime_basis(f::Nedelec, ::Quadrilateral) = tensor_vector_primes(f.order, (f.order, f.order + 1), (f.order + 1, f.order))

"Interior moment test functions as `(component, i, j)` of `P_i(x) P_j(y) e_component`."
function interior_moments(f::Union{RaviartThomas, Nedelec}, shape::Triangle)
    return [(c, i, j) for (i, j) in scalar_indices(f.order - 1, shape) for c in 1:2]
end

function interior_moments(f::Union{RaviartThomas, Nedelec}, ::Quadrilateral)
    k = f.order
    # RT tests (Q_{k-1,k}, Q_{k,k-1}); Nédélec tests (Q_{k,k-1}, Q_{k-1,k})
    d1, d2 = f isa RaviartThomas ? ((k - 1, k), (k, k - 1)) : ((k, k - 1), (k - 1, k))
    return [[(1, i, j) for j in 0:d1[2] for i in 0:d1[1]];
            [(2, i, j) for j in 0:d2[2] for i in 0:d2[1]]]
end

edge_weight(::RaviartThomas, shape, i) = edge_normal(shape, i)
edge_weight(::Nedelec, shape, i) = edge_tangent(shape, i)

function dof_functionals(f::Union{RaviartThomas, Nedelec}, shape::ReferenceShape)
    k = f.order
    functionals = DofFunctional{2}[]
    if k > 0
        rule = quadrature_rule(shape, 2k + 2)
        for (c, i, j) in interior_moments(f, shape)
            unit = c == 1 ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
            weights = [w * legendre_product_value(i, j, ξ) * unit for (ξ, w) in zip(rule.points, rule.weights)]
            push!(functionals, DofFunctional(rule.points, weights))
        end
    end
    t, _ = gauss_legendre(k + 1)
    for i in eachindex(reference_vertices(shape)), m in 1:(k + 1)
        push!(functionals, DofFunctional([edge_point(shape, i, t[m])], [edge_weight(f, shape, i)]))
    end
    return functionals, SVector{2, Float64}[]
end

# --- Brezzi–Douglas–Marini on quadrilaterals ----------------------------------------

"""
    BrezziDouglasMarini(k)

BDM_k on quadrilaterals, `k ≥ 1`: `[P_k]² ⊕ span{curl(x^{k+1} y), curl(x y^{k+1})}`
(dimension `(k + 1)(k + 2) + 2`, 8 for `k = 1`), normal components at the
`k + 1` Gauss points of every edge and, for `k ≥ 2`, moments against
`[P_{k-2}]²`. Same normal traces as `RaviartThomas(k)` with fewer dofs.
"""
struct BrezziDouglasMarini <: ElementFamily
    order::Int
    BrezziDouglasMarini(k::Integer) = k >= 1 ? new(k) : throw(ArgumentError("BDM order must be at least 1"))
end

function_space(::BrezziDouglasMarini) = :Hdiv
value_size(::BrezziDouglasMarini) = 2

function (f::BrezziDouglasMarini)(depth::Integer, n::Integer)
    depth == 0 && return 0
    depth == 1 && return f.order + 1
    n == QUADRILATERAL && return f.order * (f.order - 1)
    unsupported(f, n)
end

"Legendre coefficients of the monomial `x^a y^b` with maximal degree `D`."
monomial(a, b, D) = foldl((c, _) -> mul_y(c), 1:b; init=foldl((c, _) -> mul_x(c), 1:a; init=legendre_product(0, 0, D)))

function bdm_primes(k)
    D = k + 1
    functions = Vector{Vector{Matrix{Float64}}}()
    for (i, j) in scalar_indices(k, Triangle())  # all P_i P_j with i + j ≤ k span P_k
        push!(functions, [legendre_product(i, j, D), zero_coeffs(D)], [zero_coeffs(D), legendre_product(i, j, D)])
    end
    # curl ψ = (∂ψ/∂y, -∂ψ/∂x) for ψ = x^{k+1} y and ψ = x y^{k+1}
    push!(functions, [monomial(k + 1, 0, D), -(k + 1) * monomial(k, 1, D)])
    push!(functions, [(k + 1) * monomial(1, k, D), -monomial(0, k + 1, D)])
    return pack_primes(functions, D, 2)
end

prime_basis(f::BrezziDouglasMarini, ::Quadrilateral) = bdm_primes(f.order)
prime_basis(f::BrezziDouglasMarini, shape::Triangle) = unsupported(f, 3)
interior_moments(f::BrezziDouglasMarini, ::Quadrilateral) =
    [(c, i, j) for (i, j) in scalar_indices(f.order - 2, Triangle()) for c in 1:2]
edge_weight(::BrezziDouglasMarini, shape, i) = edge_normal(shape, i)

function dof_functionals(f::BrezziDouglasMarini, shape::Quadrilateral)
    k = f.order
    functionals = DofFunctional{2}[]
    if k >= 2
        rule = quadrature_rule(shape, 2k + 2)
        for (c, i, j) in interior_moments(f, shape)
            unit = c == 1 ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
            push!(functionals, DofFunctional(rule.points,
                                             [w * legendre_product_value(i, j, ξ) * unit for (ξ, w) in zip(rule.points, rule.weights)]))
        end
    end
    t, _ = gauss_legendre(k + 1)
    for i in 1:4, m in 1:(k + 1)
        push!(functionals, DofFunctional([edge_point(shape, i, t[m])], [edge_normal(shape, i)]))
    end
    return functionals, SVector{2, Float64}[]
end
