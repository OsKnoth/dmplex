#!/bin/bash
# GPU runs on Levante (DKRZ), one A100 per MPI rank. Submit from the repository root:
#   sbatch --account=<project> examples/gpu/levante.sbatch
# Setup (once): see docs/levante.md.
#SBATCH --job-name=dmplex-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=dmplex-gpu-%j.log

# the modules and paths of docs/levante.md (OpenMPI, ADIOS2 with CUDA, cuDSS communication layer)
source examples/gpu/levante_env.sh

export JULIA_CUDA_MEMORY_POOL=none   # needed for CUDA-aware MPI with device buffers

# kite model on GPUs, vorticity written by ADIOS2 straight from the device
srun julia --project=examples/gpu examples/galewsky.jl hexagon 6 6 --gpu --adios

# cuDSS mass solves of RT1, DG1, CG2: one GPU per rank and distributed over all ranks
srun julia --project=examples/gpu examples/gpu/cudss_mass.jl cubed 64 1
