# GPU runs on Levante

How to run the kite model on GPUs with ADIOS2 output, and the cuDSS mass solvers
on the classical element pairs, on the GPU nodes of Levante (DKRZ, 4 × A100 per node).
Everything is MPI (MPI.jl): one rank per GPU, no `Distributed`.

## What runs where

| Part | GPU path | Needs |
|---|---|---|
| `NonlinearKiteShallowWater(...; backend=CUDABackend())` | state, work vectors, metric terms, dof tables, block-diagonal mass solves (`BlockSolver`), row products, halo pack/unpack | CUDA.jl |
| `LinearShallowWater` / `NonlinearShallowWater(...; backend=CUDABackend(), solver=:cudss)` | state, operator rows (`RowMatrix`), nonlinear terms, energy and enstrophy (kernels over `CellTables`), DG solves (`BlockSolver`), RT and CG solves with cuDSS | CUDA.jl, CUDSS.jl |
| Halo exchange | device buffers passed to MPI | CUDA-aware OpenMPI (`MPI.has_gpu()`), otherwise staged through host buffers automatically |
| `vtx_writer` / `write_vtx!` | cell data handed to ADIOS2 by device pointer (`adios2_set_memory_space(gpu)`) | ADIOS2 ≥ 2.9 built with CUDA (`ADIOS2_HAVE_GPU_SUPPORT`), via `JULIA_ADIOS2_PATH` |
| `cudss_solver(M)` | Cholesky of a global mass matrix on one GPU | CUDSS.jl (cuDSS 1.x) |
| `distributed_cudss_solver(M, d, owner, comm)` | Cholesky over all ranks, rows distributed (cuDSS MGMN) | cuDSS OpenMPI communication layer (`CUDSS_COMM_LIB`) |

The mesh, the assembly of the matrices and the setup of the distribution are on
the host (the global mesh is still built on every rank). Gathered states,
diagnostics on gathered states and the VTKHDF output stay on the host.

## Setup (once)

All commands from the repository root; `examples/gpu` is a separate Julia
environment with dmplex (as a path source) plus CUDA, CUDSS, ADIOS2 and MPI,
which are only weak dependencies of dmplex.

1. **Julia** 1.12 (juliaup in `$HOME`).
2. **OpenMPI**: load a CUDA-aware OpenMPI module and make MPI.jl use it:
   ```sh
   module load openmpi        # check `module avail openmpi`
   julia --project=examples/gpu -e 'using Pkg; Pkg.instantiate(); using MPIPreferences; MPIPreferences.use_system_binary()'
   ```
   On a GPU node, `julia --project=examples/gpu -e 'using MPI, CUDA; @show MPI.has_gpu()'` should give `true`.
   Set `JULIA_CUDA_MEMORY_POOL=none` for device buffers in MPI.
3. **ADIOS2 with CUDA**, built against the same OpenMPI (spack or CMake with
   `-DADIOS2_USE_CUDA=ON -DADIOS2_USE_MPI=ON`). Point ADIOS2.jl at it:
   ```sh
   export JULIA_ADIOS2_PATH=/path/to/adios2-prefix
   julia --project=examples/gpu -e 'using Pkg; Pkg.build("ADIOS2")'
   grep ADIOS2_HAVE_GPU_SUPPORT $JULIA_ADIOS2_PATH/include/adios2/common/ADIOSConfig.h
   ```
   Without GPU support, writing a device array fails with a message saying so;
   host arrays still work with any ADIOS2.
4. **cuDSS communication layer** for the distributed solver: cuDSS ships
   `libcudss_commlayer_openmpi.so` (and the source `cudss_commlayer_openmpi.cu`
   with `cudss_build_commlayer.sh`). It must match the OpenMPI of step 2; if the
   prebuilt one does not, build it with that OpenMPI. Then
   `export CUDSS_COMM_LIB=/path/to/libcudss_commlayer_openmpi.so`.

Put the module and the two paths into `examples/gpu/levante_env.sh`.

## Runs

```sh
sbatch --account=<project> examples/gpu/levante.sbatch
```

runs, with 4 ranks on one node:

- `examples/galewsky.jl hexagon 6 6 --gpu --adios`: Galewsky jet on the kite grid
  of a hexagonal mesh, vorticity every 0.6 days into `galewsky_hexagon_6.bp`
  (open in ParaView with the *ADIOS2 VTX* reader; every rank writes its owned
  kites as one block).
- `examples/gpu/cudss_mass.jl cubed 64 1`: mass solves of RT1, DG1 and CG2 on the
  cubed sphere, one GPU per rank and distributed, with errors and timings.

## To check on the first run

These parts could not be run without NVIDIA GPUs and are written from the
cuDSS and ADIOS2 documentation:

- cuDSS 1.0 communicator: the solver sets the MPI communicator with both
  `CUDSS_DATA_COMM_DEVICE` and `CUDSS_DATA_COMM_HOST` (pointer to the
  `MPI_Comm`, size `sizeof(MPI_Comm)`). The official MGMN sample still uses a
  single `CUDSS_DATA_COMM`.
- cuDSS distributed CSR: global `nrows`/`ncols`, the local `nnz` and local
  0-based row offsets, global column indices, upper view, rows set with
  `cudssMatrixSetDistributionRow1d` (0-based, inclusive), as in the
  `simple_mgmn_distributed_matrix` sample.
- ADIOS2 device pointers: `put!` with the raw device pointer after
  `adios2_set_memory_space(var, 2)`.
