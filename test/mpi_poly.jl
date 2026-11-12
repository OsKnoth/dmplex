# Run with mpiexec: distributed polyhedral shallow water on one layer of prisms; rank 0 writes the
# change of the mass and the gathered state after a few steps of a perturbed Williamson case 2 to
# the file given as first argument; the second argument is the mesh (hexagons, triangles, cubed).
using MPI
using LinearAlgebra
using StaticArrays
using dmplex
MPI.Init()
comm = MPI.COMM_WORLD
mesh = Dict("hexagons" => create_hexagon_mesh(3), "triangles" => create_spherical_mesh(3),
            "cubed" => create_cubed_sphere_mesh(8))[ARGS[2]]
cx = extruded_complex(ExtrudedPlex(mesh, 1), [0.0, 1000.0]; radius=6.37122e6)
m = PolyShallowWater(cx; comm)
velocity(x) = (r = normalize(x); 38 * cross(SVector(0.0, 0.0, 1.0), r) + 5 * cross(SVector(1.0, 0.0, 0.0), r))
height(x) = (r = normalize(x); 1e4 - 2000 * r[3]^2 + 50 * r[1] * r[2])
y = initial_state(m, velocity, height)
mass0, dt = total_mass(m, y), shallow_water_time_step(m, y)
cache = SSPRK3Cache(y)
for _ in 1:10
    ssprk3_step!(y, dt, tendency!, cache, m)
end
mass, state = total_mass(m, y), gather_state(m, y)
if MPI.Comm_rank(comm) == 0
    open(ARGS[1], "w") do io
        println(io, (mass - mass0) / mass0, " ", length(state))
        foreach(x -> println(io, x), state)
    end
end
MPI.Finalize()
