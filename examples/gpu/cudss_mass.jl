# Mass-matrix solves of the classical element pairs (RT_k, DG_k, CG_{k+1}) on
# the cubed sphere or icosahedral grid with cuDSS: one GPU per rank, and the
# row-distributed matrix over all MPI ranks (multi-GPU multi-node mode).
# Every solve is checked against the known solution of `M x = b`.
#
#   julia --project=examples/gpu examples/gpu/cudss_mass.jl [mesh] [resolution] [k]
#   srun -n 8 julia --project=examples/gpu examples/gpu/cudss_mass.jl cubed 64 1
#
# mesh: cubed | icosahedron. The distributed solve needs the cuDSS OpenMPI
# communication layer (CUDSS_COMM_LIB) and MPI.jl on the same OpenMPI.
using CUDA
using CUDSS
using KernelAbstractions
using LinearAlgebra
using MPI
using Printf
using Random
using SparseArrays
using dmplex

function main(args)
    mesh = length(args) >= 1 ? args[1] : "cubed"
    resolution = length(args) >= 2 ? parse(Int, args[2]) : 16
    k = length(args) >= 3 ? parse(Int, args[3]) : 1

    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    local_rank = MPI.Comm_rank(MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank))
    CUDA.device!(local_rank % length(CUDA.devices()))
    backend = CUDABackend()
    root = rank == 0

    dm = mesh == "cubed" ? create_cubed_sphere_mesh(resolution) : create_spherical_mesh(resolution)
    cells = CellDistribution(dm, comm)
    root && @printf("%d cells, %d ranks, MPI CUDA-aware: %s\n", num_cells(dm), nranks, MPI.has_gpu())
    for family in (RaviartThomas(k), Discontinuous(k), Lagrange(k + 1))
        V = FESpace(dm, family)
        M = assemble_mass(V)
        n = num_dofs(V)
        x_true = randn(Xoshiro(1), n)          # the same on every rank
        b = M * x_true

        # one GPU per rank, the whole matrix
        t_factor = @elapsed S = cudss_solver(M)
        x = CUDA.zeros(Float64, n)
        ldiv!(x, S, CuVector(b))
        t_solve = @elapsed (ldiv!(x, S, CuVector(b)); CUDA.synchronize())
        error_single = maximum(abs, Array(x) - x_true)

        # all ranks, rows distributed
        d, owner = DofDistribution(V, cells)
        dd = to_backend(backend, d)
        b_local, x_local = to_backend(backend, b[d.global_ids]), CUDA.zeros(Float64, num_local(d))
        MPI.Barrier(comm)
        t_dfactor = @elapsed Sd = distributed_cudss_solver(M, dd, owner, comm)
        ldiv!(x_local, Sd, b_local)
        MPI.Barrier(comm)
        t_dsolve = @elapsed ldiv!(x_local, Sd, b_local)   # includes the halo exchange of the ghosts
        error_local = maximum(abs, Array(x_local) - x_true[d.global_ids]; init=0.0)
        error_distributed = MPI.Allreduce(error_local, max, comm)
        t_dfactor, t_dsolve = MPI.Allreduce(t_dfactor, max, comm), MPI.Allreduce(t_dsolve, max, comm)
        root && @printf("%-28s %8d dofs, nnz %9d | 1 GPU: error %.1e, factor %.3f s, solve %.2f ms | %d GPUs: error %.1e, factor %.3f s, solve %.2f ms\n",
                        string(family), n, nnz(M), error_single, t_factor, 1000t_solve, nranks, error_distributed,
                        t_dfactor, 1000t_dsolve)
    end
end

main(ARGS)
