#!/bin/bash
# GPU runs on Levante (DKRZ), one A100 per MPI rank. Submit from the repository root:
#   sbatch --account=bb1143 examples/gpu/levante.sbatch
# Setup (once): see docs/levante.md.
#SBATCH --job-name=dmplex-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=dmplex-gpu-%j.log

source examples/gpu/levante_env.sh

# cuDSS mass solves of RT1, DG1 and CG2: one GPU per rank and distributed over all ranks
srun --mpi=pmix julia --project=examples/gpu examples/gpu/cudss_mass.jl cubed 64 1

# classical RT1 × DG1 × CG2 model (Williamson TC2), coupled mass matrices with cuDSS
srun --mpi=pmix julia --project=examples/gpu examples/nonlinear_shallow_water.jl 1 cubed 64 --gpu

# kite model on GPUs (add --adios with a CUDA-enabled ADIOS2, see docs/levante.md)
srun --mpi=pmix julia --project=examples/gpu examples/galewsky.jl hexagon 6 6 --gpu
