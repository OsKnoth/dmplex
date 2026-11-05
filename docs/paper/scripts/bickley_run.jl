#
#   julia --project bickley_run.jl quadrilateral 64 40
using Serialization, dmplex
include(joinpath(@__DIR__, "..", "..", "..", "examples", "bickley.jl"))
cells, n, T = ARGS[1], parse(Int, ARGS[2]), parse(Float64, ARGS[3])
model, y = main([cells, string(n), string(T)])
ζ, _, _ = diagnostics(model, y[velocity_dofs(model)], y[height_dofs(model)])
serialize(joinpath(@__DIR__, "data", "bickley_$(cells)$(n).jls"), (ζ=ζ, n=n, T=T, nkites=num_cells(model.kites)))
