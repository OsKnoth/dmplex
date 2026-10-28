# Run with mpiexec: distributed linear and nonlinear shallow water (RT1 × DG1 (× CG2))
# on triangles and quadrilaterals; rank 0 writes, per run, the changes of mass and
# energy and the gathered state after a few steps to the file given as first argument.
using MPI
using StaticArrays
using dmplex
MPI.Init()
comm = MPI.COMM_WORLD
velocity(x) = 0.2 * SVector(-x[2], x[1], 0.0) + 0.05 * SVector(0.0, x[3], -x[2])
height(x) = 1 - 0.22 * x[3]^2 + 0.05 * x[1]
runs = [(model, dm) for model in (LinearShallowWater, NonlinearShallowWater)
                    for dm in (create_spherical_mesh(1), create_cubed_sphere_mesh(3))]
results = map(runs) do (model, dm)
    m = model(dm, 1; comm)
    y = initial_state(m, velocity, height)
    mass0, energy0 = total_mass(m, y), total_energy(m, y)
    cache = RK4Cache(y)
    for _ in 1:5
        rk4_step!(y, 2e-3, tendency!, cache, m)
    end
    (total_mass(m, y) - mass0, (total_energy(m, y) - energy0) / energy0, gather_state(m, y))
end
if MPI.Comm_rank(comm) == 0
    open(ARGS[1], "w") do io
        for (dmass, denergy, state) in results
            println(io, dmass, " ", denergy, " ", length(state))
            foreach(x -> println(io, x), state)
        end
    end
end
MPI.Finalize()
