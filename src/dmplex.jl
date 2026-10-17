module dmplex

include("DMPlexModule.jl")
include("SectionModule.jl")
include("SphereMesh.jl")
include("CubedSphereMesh.jl")
include("HexagonMesh.jl")
include("MetricModule.jl")
include("DataInitModule.jl")
include("AdvectionModule.jl")
include("TimeStepperModule.jl")
include("OutputModule.jl")

using Reexport: @reexport
@reexport using .DMPlexModule
@reexport using .SectionModule
@reexport using .SphereMesh
@reexport using .CubedSphereMesh
@reexport using .HexagonMesh
@reexport using .MetricModule
@reexport using .DataInitModule
@reexport using .AdvectionModule
@reexport using .TimeStepperModule
@reexport using .OutputModule

end # module dmplex
