# Galewsky jet with the kite shallow-water model, saves the vorticity coefficients
#
#   julia --project galewsky_run.jl hexagon 5 6
using Serialization, dmplex
include(joinpath(@__DIR__, "..", "..", "..", "examples", "galewsky.jl"))
mesh, res, days = ARGS[1], parse(Int, ARGS[2]), parse(Float64, ARGS[3])
model, y = main([mesh, string(res), string(days), "true"])
ζ = mass_solver(model.Z; lumped=true) \ (-(model.R * y[global_velocity_dofs(model)]))
serialize(joinpath(@__DIR__, "data", "galewsky_$(mesh)$(res)_day$(round(Int, days)).jls"),
          (mesh=mesh, res=res, days=days, ζ=ζ, nprimal=num_cells(model.primal), nkites=num_cells(model.kites)))
