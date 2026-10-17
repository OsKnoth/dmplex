"""
    MassSolver

Assembled mass matrix of a space together with the means to invert it:
a sparse Cholesky factorization in general, or the inverses of the cell
blocks when every dof belongs to a single cell (discontinuous spaces), which
makes the solve a cheap cell-local operation. Apply it with `S \\ b` or
`ldiv!(x, S, b)`.
"""
struct MassSolver{F}
    matrix::SparseMatrixCSC{Float64, Int}
    factor::F
end

"Inverted cell blocks of a block-diagonal mass matrix."
struct CellBlockInverse
    indices::Vector{Vector{Int}}
    inverses::Vector{Matrix{Float64}}
end

function MassSolver(V::FESpace; degree::Integer=default_degree(V))
    M = assemble_mass(V; degree)
    if function_space(V) == :L2
        indices = [first(cell_dofs(V, c)) for c in cell_range(V.dm)]
        return MassSolver(M, CellBlockInverse(indices, [inv(Matrix(M[i, i])) for i in indices]))
    end
    return MassSolver(M, cholesky(Symmetric(M)))
end

"""
    mass_solver(V; degree)

Mass matrix and factorization of `V`, assembled once and cached in the space,
so all models and operators built on `V` share them.
"""
mass_solver(V::FESpace; degree::Integer=default_degree(V)) =
    get!(() -> MassSolver(V; degree), V.operators, (:mass, Int(degree)))::MassSolver

function LinearAlgebra.ldiv!(x::AbstractVector, S::MassSolver{CellBlockInverse}, b::AbstractVector)
    for (i, Minv) in zip(S.factor.indices, S.factor.inverses)
        x[i] = Minv * b[i]
    end
    return x
end

LinearAlgebra.ldiv!(x::AbstractVector, S::MassSolver, b::AbstractVector) = copyto!(x, S.factor \ b)

Base.:\(S::MassSolver, b::AbstractVector) = ldiv!(similar(b, Float64), S, b)
