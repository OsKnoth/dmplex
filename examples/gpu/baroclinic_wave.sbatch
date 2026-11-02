#!/bin/bash
# Baroclinic wave with the nonhydrostatic TRiSK model on one A100 of Levante (DKRZ).
# Submit from the repository root:
#   sbatch --account=bb1143 examples/gpu/baroclinic_wave.sbatch
# Setup (once): see docs/levante.md.
#SBATCH --job-name=dmplex-bw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=dmplex-bw-%j.log

source examples/gpu/levante_env.sh

# hexagon 6 (≈ 120 km) for comparison with the CPU runs, then hexagon 7 (≈ 60 km)
julia --project=examples/gpu examples/baroclinic_wave.jl 6 10 300 30 --gpu
julia --project=examples/gpu examples/baroclinic_wave.jl 7 10 150 30 --gpu
