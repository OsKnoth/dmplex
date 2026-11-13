# Galewsky et al. (2004) on one layer of prisms over the sphere (PolyShallowWater), MPI and GPU
#   MESH=icosahedral|hexagonal|cubed LEVEL=5 DAYS=6 CV=auto|perot|linear PERTURB=1 GPU=0|1 OUT=file
using MPI, dmplex, LinearAlgebra, StaticArrays, Printf, KernelAbstractions
const USE_GPU = get(ENV, "GPU", "0") == "1"
USE_GPU && using CUDA
MPI.Init(); comm = MPI.COMM_WORLD; rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
backend = KernelAbstractions.CPU()
if USE_GPU
    local_rank = MPI.Comm_rank(MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank))
    CUDA.device!(local_rank % length(CUDA.devices())); backend = CUDA.CUDABackend()
end
a, Ω, g, H = 6.37122e6, 7.292e-5, 9.80616, 1.0e4
umax, φ0, φ1 = 80.0, π / 7, π / 2 - π / 7
en = exp(-4 / (φ1 - φ0)^2)
jet(φ) = φ0 < φ < φ1 ? umax / en * exp(1 / ((φ - φ0) * (φ - φ1))) : 0.0
lat(x) = asin(clamp(normalize(x)[3], -1, 1)); lon(x) = atan(x[2], x[1])
uvec(x) = (r = hypot(x[1], x[2]); r < 1e-8 * norm(x) ? zero(SVector{3, Float64}) : jet(lat(x)) * SVector(-x[2] / r, x[1] / r, 0.0))
# g h(φ) = g h₀ - ∫ a u (2Ω sin φ + tan φ u / a) dφ, h₀ such that the mean depth is H
function balanced()
    φ = range(-π / 2, π / 2; length=40001)
    f = [a * jet(p) * (2Ω * sin(p) + tan(p) * jet(p) / a) for p in φ]; f[[1, end]] .= 0
    prof = -[0.0; cumsum((f[1:end-1] .+ f[2:end]) ./ 2 .* step(φ))] ./ g
    mean = sum((prof[1:end-1] .+ prof[2:end]) ./ 2 .* (cos.(φ[1:end-1]) .+ cos.(φ[2:end])) ./ 2 .* step(φ)) / 2
    return p -> (s = (p + π / 2) / step(φ) + 1; i = clamp(floor(Int, s), 1, length(φ) - 1);
                 H - mean + prof[i] + (s - i) * (prof[i + 1] - prof[i]))
end
hb = balanced()
pert(x) = (λ = lon(x); φ = lat(x); 120cos(φ) * exp(-(3λ)^2) * exp(-(15 * (π / 4 - φ))^2))
meshes = Dict("icosahedral" => create_spherical_mesh, "hexagonal" => create_hexagon_mesh, "cubed" => create_cubed_sphere_mesh)
mesh, level, days = get(ENV, "MESH", "icosahedral"), parse(Int, get(ENV, "LEVEL", "4")), parse(Float64, get(ENV, "DAYS", "6"))
perturb = get(ENV, "PERTURB", "1") == "1"
t0 = time()
cx = extruded_complex(ExtrudedPlex(meshes[mesh](level), 1), [0.0, 1000.0]; radius=a)
m = PolyShallowWater(cx; gravity=g, omega=Ω, cell_vector=Symbol(get(ENV, "CV", "auto")),
                     comm=nranks > 1 ? comm : nothing, backend)
geo = m.geometry; nc = num_poly_cells(cx)
hfun(x) = hb(lat(x)) + (perturb ? pert(x) : 0.0)
y = initial_state(m, uvec, hfun)
dt = shallow_water_time_step(m, y; cfl=0.4)
n = ceil(Int, days * 86400 / dt); dt = days * 86400 / n
rank == 0 && @printf("%s level %d %s: %d cells, %d ranks, %s, dt %.1f s, %d steps, setup %.0f s\n", mesh, level, m.cell_vector,
                     nc, nranks, USE_GPU ? "GPU" : "CPU $(Threads.nthreads()) threads", dt, n, time() - t0)
flush(stdout)
h0 = [hfun(x) for x in geo.cell_centroid]
cache = SSPRK3Cache(y); t0 = time(); mass0 = total_mass(m, y)
for s in 1:n
    ssprk3_step!(y, dt, tendency!, cache, m)
    if s % round(Int, 86400 / dt) == 0 || s == n
        state = gather_state(m, y); h = state[1:nc]; mass = total_mass(m, y)
        if !all(isfinite, h)
            rank == 0 && println("blown up by day ", s * dt / 86400); break
        end
        rank == 0 && @printf("day %4.1f  h %.1f..%.1f  |h - h0|max %.2f m  mass %.1e  %.0f s\n", s * dt / 86400, extrema(h)...,
                             maximum(abs, h .- h0), abs(mass / mass0 - 1), time() - t0)
        flush(stdout)
    end
end
# relative vorticity at the cells: Stokes with the mean of the two cell velocities at every face
state = gather_state(m, y); h = state[1:nc]
M = Array(cell_momentum!(m, y))
mc_global = [dmplex.Parallel.gather(m.cells, M[k, :], nc, nranks > 1 ? comm : nothing) for k in 1:3]
if rank == 0
    fc = face_cells(cx)
    mc = [SVector(mc_global[1][c], mc_global[2][c], mc_global[3][c]) / h[c] for c in 1:nc]
    ζ = zeros(nc)
    for f in findall(m.interior)
        c1, c2 = fc[1, f], fc[2, f]
        c = geo.area[f] * dot((mc[c1] + mc[c2]) / 2, cross(normalize(geo.face_centroid[f]), geo.normal[f]))
        ζ[c1] += c; ζ[c2] -= c
    end
    ζ ./= geo.volume
    @printf("ζ min/max %.2e %.2e\n", extrema(ζ)...)
    out = get(ENV, "OUT", "")
    isempty(out) || open(out, "w") do io
        for (x, z, hc) in zip(geo.cell_centroid, ζ, h)
            println(io, lon(x), " ", lat(x), " ", z, " ", hc)
        end
    end
end
