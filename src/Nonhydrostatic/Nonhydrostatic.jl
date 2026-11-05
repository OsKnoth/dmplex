"""
Nonhydrostatic models on extruded meshes: the dry compressible Euler equations
with three horizontal discretizations on the same vertical grid and column
solver (`Columns.jl`):

- `ICON/`: `ICONNonhydrostatic`, the triangular C grid and the
  predictor–corrector time stepping of ICON (Zängl et al. 2015);
- `TRiSK/`: `TRiSKNonhydrostatic`, the hexagonal C grid with the TRiSK
  operators in the forms of ICON-IAP and MPAS, Wicker–Skamarock RK3 (HEVI);
- `Kite/`: `KiteNonhydrostatic`, the kite finite elements as a direct product
  of the 2D kite spaces and the layers (in progress, `docs/KITE3D_PLAN.md`).

Test cases: the baroclinic wave of Ullrich et al. (2014) and a steady thermal
wind (`BaroclinicWave.jl`). See `docs/dycores/`.
"""
module Nonhydrostatic

using LinearAlgebra
using StaticArrays
using KernelAbstractions
using ..Topology
import ..Topology: num_layers
using ..FiniteVolumes: TRiSKGrid, TRiSKShallowWater, trisk_pv_edges, iap_trisk_weights, num_cells, num_edges, num_vertices,
    bilinear_weights, rbf_coefficients
using ..FEM: to_backend, project, row_mul!, reference_element, cell_dofs, num_dofs, gauss_legendre, KITE_PRIMAL_FACES, tabulate
using ..Geometry: CellMap, map_point, edge_point, edge_normal
using ..ShallowWater: NonlinearKiteShallowWater, height_dofs, velocity_dofs
using ..Parallel: num_local_kites
import ..ShallowWater
import ..ShallowWater: initial_state, total_mass

export VerticalGrid, TRiSKNonhydrostatic, ICONNonhydrostatic, zero_state, stage!, step!, baroclinic_wave, baroclinic_channel, thermal_wind, lowest_pressure
export KiteNonhydrostatic, num_columns, project_layers, project_velocity, mass_divergence!, column_weights,
    column_points, vertical_stage!, vertical_step!, horizontal_tendency!

include("VerticalGrid.jl")
include("Columns.jl")
include("TRiSK/TRiSKNonhydrostatic.jl")
include("ICON/ICONNonhydrostatic.jl")
include("BaroclinicWave.jl")
include("Kite/KiteNonhydrostatic.jl")
include("Kite/KiteAdvection.jl")

end # module
