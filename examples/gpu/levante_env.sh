# Environment for GPU runs on Levante; adapt the module names and paths to the
# current software stack (`module avail openmpi`, `spack find adios2`).
module load openmpi                     # CUDA-aware OpenMPI (UCX); MPI.jl uses it via MPIPreferences
export JULIA_ADIOS2_PATH=$HOME/adios2-cuda               # ADIOS2 built with CUDA and this OpenMPI
export CUDSS_COMM_LIB=$HOME/cudss/libcudss_commlayer_openmpi.so   # cuDSS OpenMPI communication layer
