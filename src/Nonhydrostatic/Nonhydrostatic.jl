"""
Nonhydrostatic models on extruded meshes: the dry compressible Euler equations
with several horizontal discretizations on the same vertical grid:

- `Common/`: physical constants and the kernel launcher, the vertical grid, the
  column solver, the C-grid operators shared by ICON and MPAS, the
  Wicker–Skamarock RK3 driver and the test cases;
- `ICON/`: `ICONNonhydrostatic`, the triangular C grid and the
  predictor–corrector time stepping of ICON (Zängl et al. 2015);
- `MPAS/`: the TRiSK operators on Voronoi meshes (`Operators.jl`) and
  `Gassmann/`: `MPASGassmannNonhydrostatic` (formerly `TRiSKNonhydrostatic`),
  the ICON-IAP form of Gassmann (2013), Wicker–Skamarock RK3 (HEVI);
- `Kite/`: `KiteNonhydrostatic`, the kite finite elements as a direct product
  of the 2D kite spaces and the layers (`docs/KITE3D_PLAN.md`).

Test cases: the baroclinic wave of Ullrich et al. (2014), the baroclinic
channel of Ullrich, Reed and Jablonowski (2015) and a steady thermal wind
(`Common/TestCases.jl`). See `docs/dycores/`.
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

export VerticalGrid, MPASGassmannNonhydrostatic, TRiSKNonhydrostatic, ICONNonhydrostatic, zero_state, stage!, step!, baroclinic_wave, baroclinic_channel, thermal_wind, lowest_pressure
export KiteNonhydrostatic, num_columns, project_layers, project_velocity, mass_divergence!, column_weights,
    column_points, vertical_stage!, vertical_step!, horizontal_tendency!

include("Common/Constants.jl")
include("Common/VerticalGrid.jl")
include("Common/ColumnSolver.jl")
include("Common/CGridOperators.jl")
include("Common/TimeStepping.jl")
include("Common/TestCases.jl")
include("MPAS/Operators.jl")
include("MPAS/Gassmann/MPASGassmannNonhydrostatic.jl")
include("MPAS/Gassmann/Tendencies.jl")
include("MPAS/Gassmann/Diffusion.jl")
include("MPAS/Gassmann/Stepping.jl")
include("ICON/ICONNonhydrostatic.jl")
include("ICON/VelocityAdvection.jl")
include("ICON/SolveNH.jl")
include("ICON/Diffusion.jl")
include("ICON/Stepping.jl")
include("Kite/KiteNonhydrostatic.jl")
include("Kite/KiteAdvection.jl")

end # module
