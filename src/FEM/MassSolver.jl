"""
    MassSolver

Assembled mass matrix of a space together with the means to invert it. If the
matrix decouples into small independent blocks (diagonal lumped matrices,
discontinuous spaces, broken spaces with dual-cell blocks) the inverses of the
blocks are stored and a solve is a set of small local products; otherwise a
sparse Cholesky factorization is used. Apply it with `S \\ b` or
`ldiv!(x, S, b)`.
"""
struct MassSolver{F}
    matrix::SparseMatrixCSC{Float64, Int}
    factor::F
end

"Inverted diagonal blocks of a block-diagonal matrix."
struct BlockInverse
    indices::Vector{Vector{Int}}
    inverses::Vector{Matrix{Float64}}
end

"Index sets of the connected components of the sparsity graph of `M` (only rows with entries)."
function matrix_blocks(M::SparseMatrixCSC)
    n = size(M, 1)
    parent = collect(1:n)
    root(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    rows = rowvals(M)
    for j in 1:n, k in nzrange(M, j)
        a, b = root(rows[k]), root(j)
        a != b && (parent[a] = b)
    end
    blocks = Dict{Int, Vector{Int}}()
    for i in 1:n
        isempty(nzrange(M, i)) && continue
        push!(get!(blocks, root(i), Int[]), i)
    end
    return collect(values(blocks))
end

function MassSolver(V::FESpace; degree::Integer=default_degree(V), lumped::Bool=false, max_block::Integer=512)
    M = assemble_mass(V; degree, lumped)
    blocks = matrix_blocks(M)
    if maximum(length, blocks) <= max_block
        return MassSolver(M, BlockInverse(blocks, [inv(Matrix(M[i, i])) for i in blocks]))
    end
    return MassSolver(M, cholesky(Symmetric(M)))
end

"""
    mass_solver(V; degree, lumped=false)

Mass matrix and its inverse for `V`, assembled once and cached in the space,
so all models and operators built on `V` share them. `lumped=true` uses the
element nodes as quadrature (Gauss–Lobatto or Gauss–Radau tensor nodes),
which gives a diagonal matrix.
"""
mass_solver(V::FESpace; degree::Integer=default_degree(V), lumped::Bool=false) =
    get!(() -> MassSolver(V; degree, lumped), V.operators, (:mass, Int(degree), lumped))::MassSolver

function LinearAlgebra.ldiv!(x::AbstractVector, S::MassSolver{BlockInverse}, b::AbstractVector)
    for (i, Minv) in zip(S.factor.indices, S.factor.inverses)
        x[i] = Minv * b[i]
    end
    return x
end

LinearAlgebra.ldiv!(x::AbstractVector, S::MassSolver, b::AbstractVector) = copyto!(x, S.factor \ b)

Base.:\(S::MassSolver, b::AbstractVector) = ldiv!(zeros(length(b)), S, b)

"Largest block of a block-diagonal mass solver (0 for a Cholesky factorization)."
block_size(S::MassSolver{BlockInverse}) = maximum(length, S.factor.indices)
block_size(::MassSolver) = 0
