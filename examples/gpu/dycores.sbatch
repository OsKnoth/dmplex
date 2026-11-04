#!/bin/bash
# Baroclinic wave with the ICON and the TRiSK nonhydrostatic models on one A100 of Levante (DKRZ):
# minimum surface pressure per day and the time per step at comparable resolutions.
# Submit from the repository root:
#   sbatch --account=bb1143 examples/gpu/dycores.sbatch
# Setup (once): see docs/levante.md.
#SBATCH --job-name=dmplex-dycores
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=dmplex-dycores-%j.log

source examples/gpu/levante_env.sh

# ICON on triangles 6 and 7 (R2B5 ≈ 80 km, R2B6 ≈ 40 km), dt about ICON's dynamics substep
julia --project=examples/gpu examples/baroclinic_wave.jl 6 10 75 30 --gpu --icon
julia --project=examples/gpu examples/baroclinic_wave.jl 7 10 37.5 30 --gpu --icon
# TRiSK on hexagons 6 and 7 (≈ 120 km, 60 km)
julia --project=examples/gpu examples/baroclinic_wave.jl 6 10 300 30 --gpu
julia --project=examples/gpu examples/baroclinic_wave.jl 7 10 150 30 --gpu
