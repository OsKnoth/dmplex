"Shallow-water models on the sphere built on the FEM spaces."
module ShallowWater

using LinearAlgebra
using StaticArrays
using SparseArrays
using ..Topology
using ..Grids: create_kite_mesh
using ..Geometry
using ..FEM
using ..Parallel
using Atomix: @atomic
using KernelAbstractions

export LinearShallowWater, NonlinearShallowWater, LinearKiteShallowWater, NonlinearKiteShallowWater, velocity_dofs, height_dofs, initial_state, tendency!,
    total_mass, total_energy, potential_enstrophy, diagnostics, gather_state, center_values,
    global_velocity_dofs, global_height_dofs

# The interface shared by the dycores of dmplex (the nonhydrostatic models and PolyEuler), so that a
# test case can run with any of them: `initial_state(m, profile)`, `zero_state(m)`, `step!(m, y, dt, ys)`,
# `total_mass(m, y)` and the diagnostics below.
"`step!(m, y, dt, ys=zero_state(m))`: advance the state `y` of the dycore `m` by `dt`."
function step! end
"`zero_state(m)`: the work arrays (or a state of zeros) of `step!`."
function zero_state end
"`surface_pressure(m, y)`: the surface pressure (Pa) of every column."
function surface_pressure end
"`lowest_temperature(m, y)`: the temperature (K) of the lowest layer of every column."
function lowest_temperature end
"`vertical_velocity(m, y)`: the vertical velocity (m/s), any layout."
function vertical_velocity end

include("LinearShallowWater.jl")
include("NonlinearShallowWater.jl")
include("KiteShallowWater.jl")
include("NonlinearKiteShallowWater.jl")

end # module
