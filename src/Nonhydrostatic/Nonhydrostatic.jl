"""
Nonhydrostatic models on extruded meshes: the vertical grid and the dry
compressible Euler equations on the extruded C grid with the TRiSK operators
per layer (`TRiSKNonhydrostatic`), the baroclinic wave of Ullrich et al.
(2014). See `docs/EXTRUSION_PLAN.md`. The same equations on the extruded kite
mesh as a direct product of the 2D kite spaces and the layers
(`KiteNonhydrostatic`, in progress, `docs/KITE3D_PLAN.md`).
"""
module Nonhydrostatic

using LinearAlgebra
using StaticArrays
using KernelAbstractions
using ..Topology
import ..Topology: num_layers
using ..FiniteVolumes: TRiSKGrid, TRiSKShallowWater, trisk_pv_edges, iap_trisk_weights, num_cells, num_edges, num_vertices
using ..FEM: to_backend, project, row_mul!, reference_element, cell_dofs, num_dofs
using ..Geometry: CellMap, map_point
using ..ShallowWater: NonlinearKiteShallowWater, height_dofs, velocity_dofs
import ..ShallowWater: initial_state, total_mass

export VerticalGrid, TRiSKNonhydrostatic, zero_state, stage!, step!, baroclinic_wave, lowest_pressure
export KiteNonhydrostatic, num_columns, project_layers, project_velocity, mass_divergence!, column_weights,
    column_points, vertical_stage!, vertical_step!

include("VerticalGrid.jl")
include("TRiSKNonhydrostatic.jl")
include("BaroclinicWave.jl")
include("KiteNonhydrostatic.jl")

end # module
