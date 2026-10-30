"""
Nonhydrostatic models on extruded meshes: the vertical grid and the dry
compressible Euler equations on the extruded C grid with the TRiSK operators
per layer (`TRiSKNonhydrostatic`), the baroclinic wave of Ullrich et al.
(2014). See `docs/EXTRUSION_PLAN.md`.
"""
module Nonhydrostatic

using LinearAlgebra
using StaticArrays
using KernelAbstractions
using ..Topology
import ..Topology: num_layers
using ..FiniteVolumes: TRiSKGrid, TRiSKShallowWater, num_cells, num_edges, num_vertices
using ..FEM: to_backend
import ..ShallowWater: initial_state, total_mass

export VerticalGrid, TRiSKNonhydrostatic, zero_state, stage!, step!, baroclinic_wave, lowest_pressure

include("VerticalGrid.jl")
include("TRiSKNonhydrostatic.jl")
include("BaroclinicWave.jl")

end # module
