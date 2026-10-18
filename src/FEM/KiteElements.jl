# Finite elements for the staggered kite discretization. They live on a kite
# mesh (`create_kite_mesh`) and use its canonical local numbering: vertex 1 is
# the primal vertex, vertex 3 the primal cell center, edges 1 and 4 are halves
# of primal edges and edges 2 and 3 join an edge point to the center. Which
# points carry dofs depends on these labels, so the families extend
# `num_nodes`.

const KITE_HALF_EDGE = 1
const KITE_INTERIOR_EDGE = 2
const KITE_CENTER = 2

kite_vertex_type(dm::Plex, p) = get_label_value(dm, "kite_vertex_type", p)
kite_edge_type(dm::Plex, p) = get_label_value(dm, "kite_edge_type", p)

"""
    KiteRadau(k)

Height space of the kite method: Q_k on every kite with Gauss–Radau nodes that
include the end point at the primal cell center. Nodes lie in the kite
interior, on the center–edge kite edges and at the center, never on primal
edges, so the field is continuous inside a primal cell and broken across
primal edges (broken H(grad) on the primal mesh). With the Radau nodes as
quadrature the mass matrix is diagonal. `KiteRadau(0)` is constant per primal
cell.
"""
struct KiteRadau <: ElementFamily
    order::Int
    KiteRadau(k::Integer) = k >= 0 ? new(k) : throw(ArgumentError("order must be non-negative"))
end

"""
    KiteBrokenRT(k)

Velocity space of the kite method: RTCF_k on every kite with normal continuity
across halves of primal edges (inside a dual cell) and none across the
center–edge kite edges (the dual cell boundaries), i.e. broken H(div) on the
dual mesh. The dofs of the center–edge edges are stored on the kite itself,
so the mass matrix is block diagonal with one block per dual cell.
"""
struct KiteBrokenRT <: ElementFamily
    order::Int
    KiteBrokenRT(k::Integer) = k >= 0 ? new(k) : throw(ArgumentError("order must be non-negative"))
end

function_space(::KiteRadau) = :H1
function_space(::KiteBrokenRT) = :Hdiv
value_size(::KiteRadau) = 1
value_size(::KiteBrokenRT) = 2

function Topology.num_nodes(f::KiteRadau, dm::Plex, p::Integer)
    k, depth = f.order, get_depth(dm, p)
    depth == 2 && return k^2
    depth == 1 && return kite_edge_type(dm, p) == KITE_INTERIOR_EDGE ? k : 0
    return kite_vertex_type(dm, p) == KITE_CENTER ? 1 : 0
end

function Topology.num_nodes(f::KiteBrokenRT, dm::Plex, p::Integer)
    k, depth = f.order, get_depth(dm, p)
    depth == 2 && return 2k * (k + 1) + 2 * (k + 1)
    depth == 1 && return kite_edge_type(dm, p) == KITE_HALF_EDGE ? k + 1 : 0
    return 0
end

# Families are also called as `family(depth, cone_size)`; the kite families
# have no such point-independent layout.
(f::Union{KiteRadau, KiteBrokenRT})(depth::Integer, n::Integer) =
    throw(ArgumentError("$f needs a kite mesh; its layout depends on point labels"))

function check_kite_shape(shape)
    shape isa Quadrilateral || throw(ArgumentError("kite elements are defined on quadrilaterals only"))
end

closure_counts(f::KiteRadau, shape::ReferenceShape) =
    (check_kite_shape(shape); [f.order^2, 0, f.order, f.order, 0, 0, 0, 1, 0])

closure_counts(f::KiteBrokenRT, shape::ReferenceShape) =
    (check_kite_shape(shape); [2f.order * (f.order + 1) + 2 * (f.order + 1), f.order + 1, 0, 0, f.order + 1, 0, 0, 0, 0])

prime_basis(f::KiteRadau, shape::Quadrilateral) =
    pack_primes([[legendre_product(i, j, f.order)] for (i, j) in scalar_indices(f.order, shape)], f.order, 1)

function kite_radau_nodes(k::Integer)
    r, _ = gauss_radau(k + 1)  # ascending, r[k + 1] = 1 (the center side)
    interior = [SVector(r[i], r[j]) for j in 1:k for i in 1:k]
    edge2 = [SVector(1.0, r[j]) for j in 1:k]          # from the edge point up to the center
    edge3 = [SVector(r[i], 1.0) for i in k:-1:1]       # from the center to the other edge point
    return [interior; edge2; edge3; [SVector(1.0, 1.0)]]
end

function dof_functionals(f::KiteRadau, shape::Quadrilateral)
    nodes = kite_radau_nodes(f.order)
    return point_value.(nodes), nodes
end

function nodal_weights(f::KiteRadau, ::Quadrilateral, nodes)
    r, w = gauss_radau(f.order + 1)
    return [rule_weight(r, w, ξ[1]) * rule_weight(r, w, ξ[2]) for ξ in nodes]
end

prime_basis(f::KiteBrokenRT, shape::Quadrilateral) = prime_basis(RaviartThomas(f.order), shape)

function dof_functionals(f::KiteBrokenRT, shape::Quadrilateral)
    functionals, nodes = dof_functionals(RaviartThomas(f.order), shape)
    k = f.order
    ni = 2k * (k + 1)
    edge(i) = ni + (i - 1) * (k + 1) .+ (1:(k + 1))
    # store the center–edge edges (2, 3) with the cell: order interior, e2, e3, e1, e4
    order = [1:ni; edge(2); edge(3); edge(1); edge(4)]
    return functionals[order], nodes
end

prime_basis(::Union{KiteRadau, KiteBrokenRT}, shape::Triangle) = check_kite_shape(shape)

# --- further velocity spaces on the kites -----------------------------------------------

"""
    KiteBrokenBDM(k)

BDM_k on every kite, broken across the dual-cell boundaries like
`KiteBrokenRT(k)`: normal continuity across halves of primal edges, the dofs
of the interior (center–edge) edges stored on the kite. 8 dofs per kite for
`k = 1` instead of 12.
"""
struct KiteBrokenBDM <: ElementFamily
    order::Int
    KiteBrokenBDM(k::Integer) = k >= 1 ? new(k) : throw(ArgumentError("BDM order must be at least 1"))
end

"""
    KiteDualRadau(p)

Velocity space of the dual cell method of Wess, Kapidani, Codecasa and
Schöberl (JCP 2024): `Q_p × Q_p` on every kite with nodes at the tensor
Gauss–Radau points that include the primal-vertex side (the mirror image of
the `KiteRadau` nodes). On the halves of primal edges the dofs are the normal
components (shared by the two kites of the dual cell), inside the kite the
component values; there are no nodes on the dual-cell boundaries, so the field
is broken there. With the nodes as quadrature the mass matrix has small
blocks (2 × 2 per node, the nodes at a primal vertex coupled around it).
"""
struct KiteDualRadau <: ElementFamily
    order::Int
    KiteDualRadau(p::Integer) = p >= 0 ? new(p) : throw(ArgumentError("order must be non-negative"))
end

function_space(::Union{KiteBrokenBDM, KiteDualRadau}) = :Hdiv
value_size(::Union{KiteBrokenBDM, KiteDualRadau}) = 2
(f::Union{KiteBrokenBDM, KiteDualRadau})(depth::Integer, n::Integer) =
    throw(ArgumentError("$f needs a kite mesh; its layout depends on point labels"))

function Topology.num_nodes(f::KiteBrokenBDM, dm::Plex, p::Integer)
    k, depth = f.order, get_depth(dm, p)
    depth == 2 && return k * (k - 1) + 2 * (k + 1)
    depth == 1 && return kite_edge_type(dm, p) == KITE_HALF_EDGE ? k + 1 : 0
    return 0
end

function Topology.num_nodes(f::KiteDualRadau, dm::Plex, p::Integer)
    k, depth = f.order, get_depth(dm, p)
    depth == 2 && return 2k * (k + 1)
    depth == 1 && return kite_edge_type(dm, p) == KITE_HALF_EDGE ? k + 1 : 0
    return 0
end

closure_counts(f::KiteBrokenBDM, shape::ReferenceShape) =
    (check_kite_shape(shape); [f.order * (f.order - 1) + 2 * (f.order + 1), f.order + 1, 0, 0, f.order + 1, 0, 0, 0, 0])
closure_counts(f::KiteDualRadau, shape::ReferenceShape) =
    (check_kite_shape(shape); [2f.order * (f.order + 1), f.order + 1, 0, 0, f.order + 1, 0, 0, 0, 0])

prime_basis(f::KiteBrokenBDM, shape::Quadrilateral) = bdm_primes(f.order)

function dof_functionals(f::KiteBrokenBDM, shape::Quadrilateral)
    functionals, nodes = dof_functionals(BrezziDouglasMarini(f.order), shape)
    k = f.order
    ni = k * (k - 1)
    edge(i) = ni + (i - 1) * (k + 1) .+ (1:(k + 1))
    return functionals[[1:ni; edge(2); edge(3); edge(1); edge(4)]], nodes  # e2, e3 stored with the cell
end

function prime_basis(f::KiteDualRadau, shape::Quadrilateral)
    p = f.order
    functions = Vector{Vector{Matrix{Float64}}}()
    for (i, j) in scalar_indices(p, shape)
        push!(functions, [legendre_product(i, j, p), zero_coeffs(p)], [zero_coeffs(p), legendre_product(i, j, p)])
    end
    return pack_primes(functions, p, 2)
end

"Mirrored Gauss–Radau points (ascending, including -1, the primal-vertex side) and weights."
function dual_radau(n)
    x, w = gauss_radau(n)
    return -reverse(x), reverse(w)
end

function dof_functionals(f::KiteDualRadau, shape::Quadrilateral)
    p = f.order
    r, _ = dual_radau(p + 1)
    e1, e2 = SVector(1.0, 0.0), SVector(0.0, 1.0)
    interior = [[DofFunctional([SVector(r[i], r[j])], [e1]) for j in 1:(p + 1) for i in 2:(p + 1)];
                [DofFunctional([SVector(r[i], r[j])], [e2]) for j in 2:(p + 1) for i in 1:(p + 1)]]
    # edge 1 (y = -1) from the primal vertex; edge 4 (x = -1) towards the primal vertex
    edge1 = [DofFunctional([SVector(r[i], -1.0)], [edge_normal(shape, 1)]) for i in 1:(p + 1)]
    edge4 = [DofFunctional([SVector(-1.0, r[j])], [edge_normal(shape, 4)]) for j in (p + 1):-1:1]
    nodes = [SVector(r[i], r[j]) for j in 1:(p + 1) for i in 1:(p + 1)]
    return [interior; edge1; edge4], nodes
end

function nodal_weights(f::KiteDualRadau, ::Quadrilateral, nodes)
    r, w = dual_radau(f.order + 1)
    return [rule_weight(r, w, ξ[1]) * rule_weight(r, w, ξ[2]) for ξ in nodes]
end

prime_basis(::Union{KiteBrokenBDM, KiteDualRadau}, shape::Triangle) = check_kite_shape(shape)
