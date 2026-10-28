# Run with mpiexec: distributed TRiSK shallow water on the hexagon mesh; rank 0 writes
# the changes of mass and energy, the potential enstrophy and the gathered state after
# a few steps of a perturbed TC2 state to the file given as first argument; the second
# argument is the PV flux (trisk or gassmann).
using MPI
using StaticArrays
using dmplex
MPI.Init()
comm = MPI.COMM_WORLD
m = TRiSKShallowWater(create_hexagon_mesh(3); comm, pv_flux=Symbol(ARGS[2]))
velocity(x) = 0.2 * SVector(-x[2], x[1], 0.0) + 0.05 * SVector(0.0, x[3], -x[2])
height(x) = 1 - 0.22 * x[3]^2 + 0.05 * x[1]
y = initial_state(m, velocity, height)
mass0, energy0 = total_mass(m, y), total_energy(m, y)
cache = RK4Cache(y)
for _ in 1:10
    rk4_step!(y, 0.02, tendency!, cache, m)
end
mass, energy, enstrophy = total_mass(m, y), total_energy(m, y), potential_enstrophy(m, y)
state = gather_state(m, y)
if MPI.Comm_rank(comm) == 0
    open(ARGS[1], "w") do io
        println(io, mass - mass0, " ", (energy - energy0) / energy0, " ", enstrophy, " ", length(state))
        foreach(x -> println(io, x), state)
    end
end
MPI.Finalize()
