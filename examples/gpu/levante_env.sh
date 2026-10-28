# Environment for GPU runs on Levante. MPI.jl and the MPI-dependent JLLs use the
# system OpenMPI 4.1.6 through examples/gpu/LocalPreferences.toml (docs/levante.md);
# no module and no LD_LIBRARY_PATH are needed. The cuDSS OpenMPI communication
# layer of the CUDSS_jll artifact works with it (set CUDSS_COMM_LIB to use another).
export PATH=$HOME/.juliaup/bin:$PATH
export JULIA_CUDA_MEMORY_POOL=none      # device buffers in CUDA-aware MPI
export UCX_ERROR_SIGNALS=""             # let Julia handle its own signals
# export JULIA_ADIOS2_PATH=$HOME/adios2-cuda   # ADIOS2 built with CUDA, for --adios on GPUs
