"""
    DofFunctional(points, weights)

Linear functional `ℓ(φ) = Σ_q weights[q] ⋅ φ(points[q])` on reference
functions: point values, normal/tangential components or moments.
"""
struct DofFunctional{V}
    points::Vector{SVector{2, Float64}}
    weights::Vector{SVector{V, Float64}}
end

point_value(ξ::SVector{2}) = DofFunctional([ξ], [SVector(1.0)])

"""
    ReferenceElement(family, shape)

Finite element on a reference cell (Ciarlet triple). The basis is dual to the
dof functionals, `ℓ_i(φ_j) = δ_ij`, and is stored as Legendre coefficients.
Dofs come in closure order: cell interior, edges `1…n` (nodes along the
counter-clockwise traversal), vertices `1…n`. `entity_dofs` gives the dof
count of (interior, each edge, each vertex). Elements are cached.
"""
struct ReferenceElement{F<:ElementFamily, S<:ReferenceShape}
    family::F
    shape::S
    entity_dofs::NTuple{3, Int}
    coeffs::Array{Float64, 4}
    nodes::Vector{SVector{2, Float64}}  # node positions of nodal (H1/L2) elements
end

const ELEMENT_CACHE = Dict{Tuple{ElementFamily, ReferenceShape}, ReferenceElement}()

ReferenceElement(family::ElementFamily, shape::ReferenceShape) =
    get!(() -> build_element(family, shape), ELEMENT_CACHE, (family, shape))

function build_element(family::ElementFamily, shape::ReferenceShape)
    primes = prime_basis(family, shape)
    functionals, nodes = dof_functionals(family, shape)
    n = size(primes, 4)
    length(functionals) == n ||
        error("$family on $shape: $(length(functionals)) functionals for $n prime functions")

    V = zeros(n, n)
    for (i, ℓ) in enumerate(functionals)
        values, _ = evaluate_coefficients(primes, ℓ.points)
        for j in 1:n, q in eachindex(ℓ.points)
            V[i, j] += dot(ℓ.weights[q], view(values, j, :, q))
        end
    end
    cond(V) < 1e12 || error("$family on $shape: dof functionals are not unisolvent")
    coeffs = reshape(reshape(primes, :, n) / V, size(primes))  # primes * inv(V)

    nv = length(reference_vertices(shape))
    entity_dofs = (family(2, nv), family(1, 2), family(0, 0))
    entity_dofs[1] + nv * (entity_dofs[2] + entity_dofs[3]) == n ||
        error("$family on $shape: dof count does not match the section layout")
    return ReferenceElement(family, shape, entity_dofs, coeffs, nodes)
end

num_dofs(el::ReferenceElement) = size(el.coeffs, 4)

"""
    tabulate(el, points) -> (values, gradients)

Reference basis at `points`: `values[dof, component, point]` and
`gradients[dof, component, direction, point]`.
"""
tabulate(el::ReferenceElement, points::AbstractVector{<:SVector{2}}) =
    evaluate_coefficients(el.coeffs, points)

# --- helpers for prime bases ---------------------------------------------------

"Pack functions given as component coefficient matrices into a 4-array."
function pack_primes(functions, D::Integer, ncomp::Integer)
    primes = zeros(D + 1, D + 1, ncomp, length(functions))
    for (b, components) in enumerate(functions), c in 1:ncomp
        primes[:, :, c, b] .= components[c]
    end
    return primes
end

"Index pairs `(i, j)` of `P_i(x) P_j(y)` spanning P_k (triangle) or Q_k (quadrilateral)."
scalar_indices(k::Integer, ::Triangle) = [(i, j) for j in 0:k for i in 0:(k - j)]
scalar_indices(k::Integer, ::Quadrilateral) = [(i, j) for j in 0:k for i in 0:k]

legendre_product_value(i::Integer, j::Integer, ξ::SVector{2}) =
    legendre(i, ξ[1])[1][i + 1] * legendre(j, ξ[2])[1][j + 1]

"""
    lattice_nodes(k, shape; interior)

Nodes of order `k` built from Gauss–Lobatto points: tensor GLL points on the
square and the Blyth–Pozrikidis construction on the triangle, which also puts
GLL points on every edge. With `interior=true` only the inner nodes.
"""
function lattice_nodes(k::Integer, ::Quadrilateral; interior::Bool)
    k == 0 && return [SVector(0.0, 0.0)]
    g, _ = gauss_lobatto(k + 1)
    range = interior ? (2:k) : (1:(k + 1))
    return [SVector(g[i], g[j]) for j in range for i in range]
end

function lattice_nodes(k::Integer, shape::Triangle; interior::Bool)
    k == 0 && return [SVector(-1 / 3, -1 / 3)]
    g, _ = gauss_lobatto(k + 1)
    v = (g .+ 1) ./ 2
    V = reference_vertices(shape)
    lo = interior ? 1 : 0
    nodes = SVector{2, Float64}[]
    for b in lo:k, a in lo:k
        c = k - a - b
        c >= lo || continue
        λ1 = (1 + 2v[a + 1] - v[b + 1] - v[c + 1]) / 3
        λ2 = (1 + 2v[b + 1] - v[c + 1] - v[a + 1]) / 3
        λ3 = (1 + 2v[c + 1] - v[a + 1] - v[b + 1]) / 3
        push!(nodes, λ1 * V[1] + λ2 * V[2] + λ3 * V[3])
    end
    return nodes
end
