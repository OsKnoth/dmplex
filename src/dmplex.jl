module dmplex

using Reexport: @reexport

include("Topology/Topology.jl")
include("Grids/Grids.jl")
include("Geometry/Geometry.jl")
include("FEM/FEM.jl")
include("FiniteVolumes/FiniteVolumes.jl")
include("TimeSteppers/TimeSteppers.jl")
include("ShallowWater/ShallowWater.jl")
include("Outputs/Outputs.jl")

@reexport using .Topology
@reexport using .Grids
@reexport using .Geometry
@reexport using .FEM
@reexport using .FiniteVolumes
@reexport using .TimeSteppers
@reexport using .ShallowWater
@reexport using .Outputs

end # module dmplex
