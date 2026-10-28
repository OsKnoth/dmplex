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

"""
    cudss_solver(M; structure="SPD") -> CudssMassSolver

Sparse direct solver for `M` on the GPU with NVIDIA cuDSS (Cholesky of the
symmetric positive definite mass matrix). Apply it with `ldiv!(x, S, b)` on
device vectors. Needs `using CUDA, CUDSS` (package extension).
"""
function cudss_solver end
