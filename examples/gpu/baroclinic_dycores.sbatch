#!/bin/bash
# The baroclinic wave (examples/baroclinic_wave.jl) with the GPU dycores of dmplex on one GPU node of
# Levante (DKRZ): mpas, icon and kite on one A100 each. poly (PolyEuler) has no GPU version yet and
# runs on a workstation: julia -t 8 --project examples/baroclinic_wave.jl poly 5 10 300 30 --out=bw_poly
# Surface fields at days 8 and 10 as text (bw_<dycore>_dayNN.txt) for examples/plot_surface.py.
# Submit from the repository root:
#   sbatch --account=bb1143 examples/gpu/baroclinic_dycores.sbatch
# Setup (once): see docs/levante.md.
#SBATCH --job-name=dmplex-bw3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=dmplex-bw3-%j.log

source examples/gpu/levante_env.sh
mkdir -p bw
run() { tag=$1; shift; srun -n 1 --exact -c 16 --gpus=1 "$@" > bw/$tag.log 2>&1 & }
J="julia --project=examples/gpu"
# hexagon 6 ≈ 120 km, triangles 6 ≈ 80 km, kites on hexagon 4; mpas (ICON-IAP stepping) needs dt ≤ 220 s on hexagon 6
run mpas $J examples/baroclinic_wave.jl mpas 6 10 150 30 --gpu --out=bw/bw_mpas
run icon $J examples/baroclinic_wave.jl icon 6 10 75 30 --gpu --out=bw/bw_icon
run kite $J examples/baroclinic_wave.jl kite 4 10 75 30 --gpu --out=bw/bw_kite
wait
grep -h -E "columns|day 10|per step" bw/*.log
