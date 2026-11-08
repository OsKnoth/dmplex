"""
    to_backend(backend, x)

Copy of the array `x` in the memory of a KernelAbstractions `backend`
(`CPU()`, `CUDABackend()`, ...).
"""
function to_backend(backend, x::AbstractArray)
    y = KernelAbstractions.allocate(backend, eltype(x), size(x))
    copyto!(y, x)
    return y
end

"""
    BlockSolver

Block-diagonal inverse in flat arrays, solved by one kernel thread per row:
row `t` writes `x[indices[t]] = Σ_c values[coefficient[t] + c] b[indices[first[t] + c]]`
for the `count[t]` entries of its block. Built from a `MassSolver{BlockInverse}`
with `block_solver`, so the same kernel runs on the CPU and on GPUs.
"""
struct BlockSolver{I<:AbstractVector{Int}, V<:AbstractVector{Float64}}
    indices::I       # rows of all blocks, concatenated
    first::I         # start of the row's block in `indices`
    count::I         # size of the row's block
    coefficient::I   # start of the row's inverse row in `values` (minus one)
    values::V        # inverse rows, block by block
end

"Block solver of the block-diagonal mass solver `S` on `backend`."
function block_solver(S::MassSolver{BlockInverse}, backend=KernelAbstractions.CPU())
    indices, first, count, coefficient, values = Int[], Int[], Int[], Int[], Float64[]
    for (block, inverse) in zip(S.factor.indices, S.factor.inverses)
        n, start = length(block), length(indices) + 1
        for r in 1:n
            push!(first, start); push!(count, n); push!(coefficient, length(values))
            append!(values, inverse[r, :])
        end
        append!(indices, block)
    end
    return BlockSolver(map(x -> to_backend(backend, x), (indices, first, count, coefficient, values))...)
end

@kernel function block_solve_kernel!(x, @Const(b), @Const(indices), @Const(first), @Const(count),
                                     @Const(coefficient), @Const(values))
    t = @index(Global)
    @inbounds begin
        s, o = first[t] - 1, coefficient[t]
        acc = 0.0
        for c in 1:count[t]
            acc += values[o + c] * b[indices[s + c]]
        end
        x[indices[t]] = acc
    end
end

function LinearAlgebra.ldiv!(x::AbstractVector, S::BlockSolver, b::AbstractVector)
    n = length(S.indices)
    n == 0 && return x
    backend = KernelAbstractions.get_backend(S.values)
    if backend isa KernelAbstractions.CPU  # a plain loop: no launch overhead, no allocation
        @inbounds for t in 1:n
            s, o = S.first[t] - 1, S.coefficient[t]
            acc = 0.0
            for c in 1:S.count[t]
                acc += S.values[o + c] * b[S.indices[s + c]]
            end
            x[S.indices[t]] = acc
        end
    else
        block_solve_kernel!(backend)(x, b, S.indices, S.first, S.count, S.coefficient, S.values; ndrange=n)
        KernelAbstractions.synchronize(backend)
    end
    return x
end

"""
    layer_workgroup(L)

Workgroup of a kernel over (layers, points) with `L` layers: the layers of a point
in adjacent threads (up to 32, so that a warp reads one line of a layers-first
array), 256 threads in all.
"""
layer_workgroup(L::Integer) = (w = clamp(L, 1, 32); (w, 256 ÷ w))

@kernel function block_solve_layers_kernel!(X, @Const(B), @Const(indices), @Const(first), @Const(count),
                                            @Const(coefficient), @Const(values))
    k, t = @index(Global, NTuple)
    @inbounds begin
        s, o = first[t] - 1, coefficient[t]
        acc = 0.0
        for c in 1:count[t]
            acc += values[o + c] * B[k, indices[s + c]]
        end
        X[k, indices[t]] = acc
    end
end

"`ldiv!` for every row of `B` (the layers of a 3D field, layers first: `B[layer, dof]`) in one launch."
function LinearAlgebra.ldiv!(X::AbstractMatrix, S::BlockSolver, B::AbstractMatrix)
    n = length(S.indices)
    (n == 0 || size(B, 1) == 0) && return X
    backend = KernelAbstractions.get_backend(S.values)
    if backend isa KernelAbstractions.CPU
        @inbounds for t in 1:n, k in axes(B, 1)
            s, o = S.first[t] - 1, S.coefficient[t]
            acc = 0.0
            for c in 1:S.count[t]
                acc += S.values[o + c] * B[k, S.indices[s + c]]
            end
            X[k, S.indices[t]] = acc
        end
    else
        block_solve_layers_kernel!(backend, layer_workgroup(size(B, 1)))(X, B, S.indices, S.first, S.count,
                                                                        S.coefficient, S.values;
                                                                        ndrange=(size(B, 1), n))
        KernelAbstractions.synchronize(backend)
    end
    return X
end

"""
    SparseBlockSolver

Cholesky factors `L` of the blocks of a block-diagonal SPD matrix with only the
nonzeros of `L` (the pattern of the block and its fill), column by column: the
rows `indices[first[b]:first[b + 1] - 1]` of block `b`, and for its row `t` the
entries `colptr[t]:colptr[t + 1] - 1` of column `t` of `L`, the diagonal first,
at the rows `indices[rowpos[e]]`. A solve is one forward and one backward
substitution per block (and layer), with a few times fewer operations than the
product with the dense inverse when the blocks are sparse (the dual-cell blocks
of the kite velocity mass). Built with `sparse_block_solver`.
"""
struct SparseBlockSolver{I<:AbstractVector{Int}, V<:AbstractVector{Float64}}
    first::I
    indices::I
    colptr::I
    rowpos::I
    values::V
end

"""
    sparse_block_solver(blocks, matrices, backend=CPU())

`SparseBlockSolver` of the SPD blocks `matrices[b]` at the rows `blocks[b]`, in
the given order of the rows (the symbolic factorization keeps the fill).
"""
function sparse_block_solver(blocks, matrices, backend=KernelAbstractions.CPU())
    first, indices, colptr, rowpos, values = [1], Int[], Int[], Int[], Float64[]
    for (rows, A) in zip(blocks, matrices)
        n, o = length(rows), length(indices)
        P = Matrix(A .!= 0)                      # pattern of L: the block and its fill
        for j in 1:n, i in j + 1:n
            P[i, j] || continue
            for l in i + 1:n
                P[l, j] && (P[l, i] = true)
            end
        end
        L = cholesky(Symmetric(Matrix(A), :L)).L
        for j in 1:n
            push!(colptr, length(values) + 1)
            push!(rowpos, o + j); push!(values, L[j, j])
            for i in j + 1:n
                P[i, j] && (push!(rowpos, o + i); push!(values, L[i, j]))
            end
        end
        append!(indices, rows)
        push!(first, length(indices) + 1)
    end
    push!(colptr, length(values) + 1)
    return SparseBlockSolver(map(x -> to_backend(backend, x), (first, indices, colptr, rowpos, values))...)
end

num_blocks(S::SparseBlockSolver) = length(S.first) - 1

# forward and backward substitution of block `b` in layer `k`, in place in `X` (the rows of the block)
@inline function sparse_block_solve!(X, B, first, indices, colptr, rowpos, values, k, b)
    @inbounds begin
        s, e = first[b], first[b + 1] - 1
        for t in s:e
            X[k, indices[t]] = B[k, indices[t]]
        end
        for t in s:e                     # L y = b
            p = colptr[t]
            y = X[k, indices[t]] / values[p]
            X[k, indices[t]] = y
            for q in p + 1:colptr[t + 1] - 1
                X[k, indices[rowpos[q]]] -= values[q] * y
            end
        end
        for t in e:-1:s                  # Lᵀ x = y
            p = colptr[t]
            x = X[k, indices[t]]
            for q in p + 1:colptr[t + 1] - 1
                x -= values[q] * X[k, indices[rowpos[q]]]
            end
            X[k, indices[t]] = x / values[p]
        end
    end
    return nothing
end

@kernel function sparse_block_solve_kernel!(X, @Const(B), @Const(first), @Const(indices), @Const(colptr),
                                            @Const(rowpos), @Const(values))
    k, b = @index(Global, NTuple)
    sparse_block_solve!(X, B, first, indices, colptr, rowpos, values, k, b)
end

"`X = A⁻¹ B` in every layer of the fields `X`, `B` (layers first, `B[layer, dof]`; `X !== B`), one thread per layer and block."
function LinearAlgebra.ldiv!(X::AbstractMatrix, S::SparseBlockSolver, B::AbstractMatrix)
    X === B && throw(ArgumentError("in-place block solve needs distinct input and output"))
    nb = num_blocks(S)
    (nb == 0 || size(B, 1) == 0) && return X
    backend = KernelAbstractions.get_backend(S.values)
    if backend isa KernelAbstractions.CPU
        for b in 1:nb, k in axes(B, 1)
            sparse_block_solve!(X, B, S.first, S.indices, S.colptr, S.rowpos, S.values, k, b)
        end
    else
        sparse_block_solve_kernel!(backend, layer_workgroup(size(B, 1)))(X, B, S.first, S.indices, S.colptr, S.rowpos,
                                                                        S.values; ndrange=(size(B, 1), nb))
        KernelAbstractions.synchronize(backend)
    end
    return X
end

LinearAlgebra.ldiv!(x::AbstractVector, S::SparseBlockSolver, b::AbstractVector) =
    (ldiv!(reshape(x, 1, :), S, reshape(b, 1, :)); x)

"""
    HostSolver(solver, n)

A solver that exists only on the host (a sparse Cholesky factorization)
applied to vectors on any backend: `ldiv!` copies the right-hand side to the
host, solves and copies the solution back. The fallback for coupled mass
matrices on GPUs without cuDSS.
"""
struct HostSolver{S}
    solver::S
    b::Vector{Float64}
    x::Vector{Float64}
end

HostSolver(solver, n::Integer) = HostSolver(solver, zeros(n), zeros(n))

function LinearAlgebra.ldiv!(x::AbstractVector, S::HostSolver, b::AbstractVector)
    copyto!(S.b, b)
    ldiv!(S.x, S.solver, S.b)
    return copyto!(x, S.x)
end

"""
    device_solver(S::MassSolver, backend; solver=:host)

Mass solver `S` for vectors on `backend`: `S` itself on the CPU, a
`BlockSolver` for block-diagonal matrices, otherwise `cudss_solver(S.matrix)`
(`solver=:cudss`) or a `HostSolver`.
"""
function device_solver(S::MassSolver, backend; solver::Symbol=:host)
    backend isa KernelAbstractions.CPU && return S
    S.factor isa BlockInverse && return block_solver(S, backend)
    solver == :cudss && return cudss_solver(S.matrix)
    return HostSolver(S, size(S.matrix, 1))
end

"""
    RowMatrix(At, backend)

Selected rows of a sparse matrix stored transposed, one CSC column of `At` per
row (`colptr`, `rowval`, `nzval` on the backend), for `y = α A x` with one
thread per row (`row_mul!`).
"""
struct RowMatrix{I<:AbstractVector{Int}, V<:AbstractVector{Float64}}
    colptr::I
    rowval::I
    nzval::V
end

RowMatrix(At::SparseMatrixCSC, backend) =
    RowMatrix(to_backend(backend, At.colptr), to_backend(backend, At.rowval), to_backend(backend, At.nzval))

num_rows(A::RowMatrix) = length(A.colptr) - 1

@kernel function row_mul_kernel!(y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(x), α)
    j = @index(Global)
    s = 0.0
    @inbounds for k in colptr[j]:(colptr[j + 1] - 1)
        s += nzval[k] * x[rowval[k]]
    end
    @inbounds y[j] = α * s
end

"`y = α A x` for the stored rows of `A`, one thread per row (a plain loop on the CPU)."
function row_mul!(y, A::RowMatrix, x, α=1.0)
    n = num_rows(A)
    n == 0 && return y
    backend = get_backend(A.nzval)
    if backend isa KernelAbstractions.CPU
        @inbounds for j in 1:n
            s = 0.0
            for k in A.colptr[j]:(A.colptr[j + 1] - 1)
                s += A.nzval[k] * x[A.rowval[k]]
            end
            y[j] = α * s
        end
    else
        row_mul_kernel!(backend)(y, A.colptr, A.rowval, A.nzval, x, α; ndrange=n)
        KernelAbstractions.synchronize(backend)
    end
    return y
end

@kernel function row_mul_layers_kernel!(Y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(X), α)
    k, j = @index(Global, NTuple)
    s = 0.0
    @inbounds for t in colptr[j]:(colptr[j + 1] - 1)
        s += nzval[t] * X[k, rowval[t]]
    end
    @inbounds Y[k, j] = α * s
end

"""
    row_mul!(Y::AbstractMatrix, A::RowMatrix, X::AbstractMatrix, α=1)

`Y[k, j] = α (A X[k, :])_j` for every layer `k` of a field layers first and every stored row `j`
(the first `num_rows(A)` columns of `Y`), one thread per layer and row.
"""
function row_mul!(Y::AbstractMatrix, A::RowMatrix, X::AbstractMatrix, α=1.0)
    n = num_rows(A)
    (n == 0 || size(X, 1) == 0) && return Y
    backend = get_backend(A.nzval)
    if backend isa KernelAbstractions.CPU
        @inbounds for j in 1:n, k in axes(X, 1)
            s = 0.0
            for t in A.colptr[j]:(A.colptr[j + 1] - 1)
                s += A.nzval[t] * X[k, A.rowval[t]]
            end
            Y[k, j] = α * s
        end
    else
        row_mul_layers_kernel!(backend, layer_workgroup(size(X, 1)))(Y, A.colptr, A.rowval, A.nzval, X, α;
                                                                    ndrange=(size(X, 1), n))
        KernelAbstractions.synchronize(backend)
    end
    return Y
end

"""
    cudss_solver(M; structure="SPD") -> CudssMassSolver

Sparse direct solver for `M` on the GPU with NVIDIA cuDSS (Cholesky of the
symmetric positive definite mass matrix). Apply it with `ldiv!(x, S, b)` on
device vectors. Needs `using CUDA, CUDSS` (package extension).
"""
function cudss_solver end
