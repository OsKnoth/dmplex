"""
Finite-element families. A family is a callable `(depth, cone_size) -> nodes`
giving the number of dof nodes per mesh point, so it can be handed to a
`Section` directly; `ReferenceElement(family, shape)` builds its basis.
"""
abstract type ElementFamily end

"Continuous Lagrange elements of order `k ≥ 1` (P_k on triangles, Q_k on quadrilaterals)."
struct Lagrange <: ElementFamily
    order::Int
    Lagrange(k::Integer) = k >= 1 ? new(k) : throw(ArgumentError("Lagrange order must be at least 1"))
end

"Discontinuous elements of order `k ≥ 0`; all nodes belong to the cell interior."
struct Discontinuous <: ElementFamily
    order::Int
    Discontinuous(k::Integer) = k >= 0 ? new(k) : throw(ArgumentError("DG order must be non-negative"))
end

"""
Raviart–Thomas elements `RT_k`, `k ≥ 0`, divergence in P_k/Q_k (RTCF on
quadrilaterals). Edge dofs are normal components at Gauss points.
"""
struct RaviartThomas <: ElementFamily
    order::Int
    RaviartThomas(k::Integer) = k >= 0 ? new(k) : throw(ArgumentError("Raviart–Thomas order must be non-negative"))
end

"Nédélec elements of the first kind `N_k`, `k ≥ 0`; edge dofs are tangential components."
struct Nedelec <: ElementFamily
    order::Int
    Nedelec(k::Integer) = k >= 0 ? new(k) : throw(ArgumentError("Nédélec order must be non-negative"))
end

function_space(::Lagrange) = :H1
function_space(::Discontinuous) = :L2
function_space(::RaviartThomas) = :Hdiv
function_space(::Nedelec) = :Hcurl

"Number of reference value components (1 scalar, 2 vector)."
value_size(::Union{Lagrange, Discontinuous}) = 1
value_size(::Union{RaviartThomas, Nedelec}) = 2

const TRIANGLE = 3
const QUADRILATERAL = 4

unsupported(family, n) = throw(ArgumentError("$family is not defined on cells with $n edges"))

function (f::Lagrange)(depth::Integer, n::Integer)
    k = f.order
    depth == 0 && return 1
    depth == 1 && return k - 1
    k == 1 && return 0
    n == TRIANGLE && return (k - 1) * (k - 2) ÷ 2
    n == QUADRILATERAL && return (k - 1)^2
    unsupported(f, n)
end

function (f::Discontinuous)(depth::Integer, n::Integer)
    k = f.order
    depth < 2 && return 0
    k == 0 && return 1
    n == TRIANGLE && return (k + 1) * (k + 2) ÷ 2
    n == QUADRILATERAL && return (k + 1)^2
    unsupported(f, n)
end

function edge_element_nodes(f, depth, n)
    k = f.order
    depth == 0 && return 0
    depth == 1 && return k + 1
    k == 0 && return 0
    n == TRIANGLE && return k * (k + 1)
    n == QUADRILATERAL && return 2k * (k + 1)
    unsupported(f, n)
end

(f::RaviartThomas)(depth::Integer, n::Integer) = edge_element_nodes(f, depth, n)
(f::Nedelec)(depth::Integer, n::Integer) = edge_element_nodes(f, depth, n)

"`FieldLayout` of a finite-element family, for use in a `Section`."
field(family::ElementFamily; name::AbstractString="u", ncomp::Integer=1) =
    FieldLayout(String(name), function_space(family), family.order, Int(ncomp), family)

lagrange(k::Integer; name::AbstractString="u", ncomp::Integer=1) = field(Lagrange(k); name, ncomp)
discontinuous(k::Integer; name::AbstractString="h", ncomp::Integer=1) = field(Discontinuous(k); name, ncomp)
raviart_thomas(k::Integer; name::AbstractString="u") = field(RaviartThomas(k); name)
nedelec(k::Integer; name::AbstractString="u") = field(Nedelec(k); name)
