"""
Sparse direct mass-matrix solvers on NVIDIA GPUs with cuDSS: one GPU
(`cudss_solver`) and multi-GPU multi-node over MPI with a row-distributed
matrix (`distributed_cudss_solver`).
"""
module dmplexCUDSSExt

using CUDSS
using CUDSS: CUDACore, cuSPARSE
using LinearAlgebra
using MPI
using SparseArrays
using dmplex
import dmplex.FEM: cudss_solver
import dmplex.Parallel: distributed_cudss_solver

"Cholesky (or `structure`) factorization of a global matrix on one GPU, with device work vectors."
struct CudssMassSolver{S}
    solver::S
    rhs::CUDACore.CuVector{Float64}
    solution::CUDACore.CuVector{Float64}
end

function cudss_solver(M::SparseMatrixCSC; structure::AbstractString="SPD")
    A = cuSPARSE.CuSparseMatrixCSR(SparseMatrixCSC{Float64, Int32}(M))
    solver = CudssSolver(A, structure, 'F')
    rhs, solution = CUDACore.zeros(Float64, size(M, 1)), CUDACore.zeros(Float64, size(M, 1))
    cudss("analysis", solver, solution, rhs)
    cudss("factorization", solver, solution, rhs; asynchronous=false)
    return CudssMassSolver(solver, rhs, solution)
end

"Solve on the GPU; `x` and `b` may be device or host vectors (copied through the device work vectors)."
function LinearAlgebra.ldiv!(x::AbstractVector, S::CudssMassSolver, b::AbstractVector)
    copyto!(S.rhs, b)
    cudss("solve", S.solver, S.solution, S.rhs; asynchronous=false)
    return copyto!(x, S.solution)
end

"""
Distributed cuDSS solver (MGMN mode): the owned rows of the global matrix in
the contiguous numbering (upper triangle, 0-based CSR), and the dense
right-hand side and solution distributed by the same row ranges.
"""
mutable struct DistributedCudssSolver{D, C}
    handle::CUDSS.cudssHandle_t
    config::CudssConfig
    data::CudssData
    A::CUDSS.cudssMatrix_t
    X::CUDSS.cudssMatrix_t
    B::CUDSS.cudssMatrix_t
    arrays::Tuple    # device arrays referenced by the cuDSS descriptors
    rhs::CUDACore.CuVector{Float64}
    solution::CUDACore.CuVector{Float64}
    comm_handle::Base.RefValue{MPI.MPI_Comm}
    distribution::D
    comm::C
    released::Bool
end

"Destroy the cuDSS objects (once): before `MPI_Finalize`, since cuDSS frees its own communicator with the data."
function release!(S::DistributedCudssSolver)
    S.released && return nothing
    S.released = true
    foreach(CUDSS.cudssMatrixDestroy, (S.A, S.X, S.B))
    finalize(S.data)
    CUDSS.cudssDestroy(S.handle)
    return nothing
end

"Communication layer library: argument, `CUDSS_COMM_LIB`, or the OpenMPI layer next to `libcudss`."
function comm_layer_library(comm_layer)
    comm_layer !== nothing && return comm_layer
    haskey(ENV, "CUDSS_COMM_LIB") && return ENV["CUDSS_COMM_LIB"]
    candidate = joinpath(dirname(CUDSS.libcudss), "libcudss_commlayer_openmpi.so")
    isfile(candidate) && return candidate
    error("no cuDSS communication layer found: set CUDSS_COMM_LIB to libcudss_commlayer_openmpi.so " *
          "(shipped with cuDSS, built against the OpenMPI that MPI.jl uses)")
end

function distributed_cudss_solver(M::SparseMatrixCSC, d::DofDistribution, owner::AbstractVector{<:Integer}, comm;
                                  structure::AbstractString="SPD", comm_layer=nothing)
    n, n_owned = size(M, 1), d.n_owned
    T, first_row = distributed_rows(M, d, owner, MPI.Comm_rank(comm))   # rows first_row + 1 : first_row + n_owned
    rowptr = CUDACore.CuVector{Int32}(T.colptr .- 1)
    colval = CUDACore.CuVector{Int32}(T.rowval .- 1)
    nzval = CUDACore.CuVector{Float64}(T.nzval)
    rhs, solution = CUDACore.zeros(Float64, n_owned), CUDACore.zeros(Float64, n_owned)

    # an own handle: the communication layer makes it a distributed (MGMN) handle, the shared
    # `CUDSS.handle()` must stay a single-GPU one
    handle = CUDSS.cudssCreate()
    CUDSS.cudssSetCommLayer(handle, comm_layer_library(comm_layer))
    config, data = CudssConfig(), CudssData(handle)
    comm_handle = Ref(comm.val)
    for param in (CUDSS.CUDSS_DATA_COMM_DEVICE, CUDSS.CUDSS_DATA_COMM_HOST)
        CUDSS.cudssDataSet(handle, data, param, comm_handle, sizeof(MPI.MPI_Comm))
    end

    last_row = first_row + n_owned - 1                       # 0-based, inclusive; empty if < first_row
    A, X, B = Ref{CUDSS.cudssMatrix_t}(), Ref{CUDSS.cudssMatrix_t}(), Ref{CUDSS.cudssMatrix_t}()
    CUDSS.cudssMatrixCreateCsr(A, n, n, length(T.nzval), rowptr, CUDACore.CU_NULL, colval, nzval,
                               Int32, Int32, Float64, structure, 'U', 'Z')
    CUDSS.cudssMatrixCreateDn(X, n, 1, n, solution, Float64, 'C')
    CUDSS.cudssMatrixCreateDn(B, n, 1, n, rhs, Float64, 'C')
    for matrix in (A, X, B)
        CUDSS.cudssMatrixSetDistributionRow1d(matrix[], first_row, last_row)
    end
    S = DistributedCudssSolver(handle, config, data, A[], X[], B[], (rowptr, colval, nzval), rhs, solution,
                               comm_handle, d, comm, false)
    finalizer(release!, S)
    MPI.add_finalize_hook!(() -> release!(S))
    execute!(S, "analysis")
    execute!(S, "factorization")
    return S
end

function execute!(S::DistributedCudssSolver, phase::AbstractString)
    CUDSS.cudssExecute(S.handle, convert(CUDSS.cudssPhase_t, phase), S.config, S.data, S.A, S.X, S.B)
    CUDACore.synchronize()
    return S
end

"""
Solve for the owned entries of the local vectors `x`, `b` (host or device) on
all ranks of the communicator; the ghosts of `x` are updated by halo exchange
(with the halo of the distribution passed at construction).
"""
function LinearAlgebra.ldiv!(x::AbstractVector, S::DistributedCudssSolver, b::AbstractVector)
    n = S.distribution.n_owned
    n > 0 && copyto!(S.rhs, 1, b, 1, n)
    execute!(S, "solve")
    n > 0 && copyto!(x, 1, S.solution, 1, n)
    exchange!(x, S.distribution, S.comm)
    return x
end

end # module
