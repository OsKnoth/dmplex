#!/bin/bash
# ICON's jabw experiment (jabw10d: R2B5, 64 levels, dtime 576 s) with ICONNonhydrostatic on one A100 of
# Levante (DKRZ), for the comparison with Fortran ICON. Submit from the repository root:
#   sbatch --account=bb1143 examples/gpu/icon_jabw.sbatch [grid file] [--init=icon.nc] [--compare=icon.nc]
# (see examples/icon_jabw.jl; --init/--compare with ICON's output of the flat dcmip_bw_11 wave)
# Setup (once): see docs/levante.md.
#SBATCH --job-name=dmplex-jabw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=dmplex-jabw-%j.log

source examples/gpu/levante_env.sh

GRID=${1:-/work/bb1143/b383021/GitHub/ICON-data/grids/0051/icon_grid_0051_R02B05_G.nc}
julia --project=examples/gpu examples/icon_jabw.jl $GRID 10 576 64 35000 --gpu "${@:2}"
