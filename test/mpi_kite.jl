# Run with mpiexec: distributed nonlinear kite model; rank 0 writes the gathered
# state after a few steps to the file given as first argument.
using MPI
using StaticArrays
using dmplex
MPI.Init()
comm = MPI.COMM_WORLD
model = NonlinearKiteShallowWater(create_hexagon_mesh(2), 1; comm)
y = initial_state(model, x -> 0.2 * SVector(-x[2], x[1], 0.0) + 0.05 * SVector(0.0, x[3], -x[2]),
                  x -> 1 - 0.22 * x[3]^2 + 0.05 * x[1])
mass0, energy0 = total_mass(model, y), total_energy(model, y)
cache = SSPRK3Cache(y)
for _ in 1:10
    ssprk3_step!(y, 2e-3, tendency!, cache, model)
end
state = gather_state(model, y)
mass, energy = total_mass(model, y), total_energy(model, y)
if MPI.Comm_rank(comm) == 0
    open(ARGS[1], "w") do io
        println(io, mass - mass0, " ", (energy - energy0) / energy0)
        foreach(x -> println(io, x), state)
    end
end
MPI.Finalize()
