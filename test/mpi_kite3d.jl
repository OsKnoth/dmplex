# Run with mpiexec: distributed 3D kite model (baroclinic wave, DG transport, tracer); rank 0 writes
# the gathered state after a few steps to the file given as first argument, the second argument
# selects the transport (dg or vector_invariant).
using MPI
using dmplex
MPI.Init()
comm = MPI.COMM_WORLD
dg = length(ARGS) < 2 || ARGS[2] == "dg"
m = KiteNonhydrostatic(create_hexagon_mesh(2), VerticalGrid(6, 30000.0), 1; damping_height=22000.0, damping_rate=0.05,
                       tracer=true, horizontal_theta=:dg_upwind, horizontal_advection=dg ? :dg : :vector_invariant,
                       comm)
y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z)); ys = zero_state(m)
for _ in 1:3
    step!(m, y, 300.0, ys)
end
state = [gather_layers(m, x) for x in y]
if MPI.Comm_rank(comm) == 0
    open(ARGS[1], "w") do io
        foreach(x -> foreach(v -> println(io, v), x), state)
    end
end
MPI.Finalize()
