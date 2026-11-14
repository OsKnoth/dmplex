# corner prototype: linear energy check and Williamson 2 against PolyShallowWater
#   EIG=1 EIGMESH=hexagonal|icosahedral|cubed EIGLEVEL=2 MESHES=hexagonal LEVELS="3 4" DAYS=5
include("corner_sw.jl")
a, Ω, g = 6.37122e6, 7.292e-5, 9.80616
u0, gh0 = 2π * a / (12 * 86400), 2.94e4
hexact(x) = (r = normalize(x); (gh0 - (a * Ω * u0 + u0^2 / 2) * r[3]^2) / g)
uexact(x) = (r = normalize(x); u0 * cross(SVector(0.0, 0.0, 1.0), r))
meshes = Dict("icosahedral" => create_spherical_mesh, "hexagonal" => create_hexagon_mesh, "cubed" => create_cubed_sphere_mesh)
mk(name, l) = extruded_complex(ExtrudedPlex(meshes[name](l), 1), [0.0, 1000.0]; radius=a)
# linear operator about rest, depth H: eigenvalues should be imaginary (energy norm skew)
if get(ENV, "EIG", "1") == "1"
    m = CornerSW(mk(get(ENV, "EIGMESH", "hexagonal"), parse(Int, get(ENV, "EIGLEVEL", "2"))))
    nc, ns = m.nc, m.ns
    H = 1.0e4
    L = zeros(nc + ns, nc + ns)
    for j in 1:nc + ns
        y = zeros(nc + ns); y[j] = 1
        dy = zeros(nc + ns)
        h, U = y[1:nc], y[nc+1:end]
        # linearize: g h²/2 → g H h'
        dy[1:nc] = -(m.D * U)
        cu = m.C * U; cor = similar(cu)
        for k in eachindex(m.ω); a_ = m.ω[k] * m.fk[k]; cor[2k-1] = a_ * cu[2k]; cor[2k] = -a_ * cu[2k-1]; end
        dy[nc+1:end] = m.Mf \ (m.B' * (g * H .* h) + m.C' * cor)
        L[:, j] = dy
    end
    λ = eigvals(L)
    @printf("%s %s: %d cells, %d subfaces; max Re λ %.2e, max |λ| %.2e\n", get(ENV, "EIGMESH", "hexagonal"), get(ENV, "EIGLEVEL", "2"), nc, ns, maximum(real, λ), maximum(abs, λ))
    fmax = 2Ω
    @printf("  |λ| < 1e-9: %d,  0 < |λ| ≤ 2Ω·1.01 (inertial-type): %d,  gravity: %d\n",
            count(x -> abs(x) < 1e-9, λ), count(x -> 1e-9 <= abs(x) <= 1.01fmax, λ), count(x -> abs(x) > 1.01fmax, λ))
end
function w2(name, l; days=5.0, corner=true)
    cx = mk(name, l)
    ref = PolyShallowWater(cx; gravity=g, omega=Ω)
    dt0 = PH.shallow_water_time_step(ref, PH.initial_state(ref, uexact, hexact); cfl=0.4)
    m = corner ? CornerSW(cx) : ref
    y = corner ? initial_state(m, uexact, hexact) : PH.initial_state(ref, uexact, hexact)
    geo = ref.geometry; nc = num_poly_cells(cx); h0 = [hexact(x) for x in geo.cell_centroid]
    n = ceil(Int, days * 86400 / dt0); dt = days * 86400 / n
    tend = corner ? tendency! : PH.tendency!
    cache = SSPRK3Cache(y); t0 = time()
    for s in 1:n
        ssprk3_step!(y, dt, tend, cache, m)
    end
    h = y[1:nc]; V = geo.volume
    return sqrt(sum(V .* (h .- h0) .^ 2) / sum(V .* h0 .^ 2)), maximum(abs, h .- h0) / maximum(abs, h0), n, time() - t0
end
for name in split(get(ENV, "MESHES", "hexagonal")), l in parse.(Int, split(get(ENV, "LEVELS", "3 4")))
    for corner in (true, false)
        l2, li, n, t = w2(name, l; days=parse(Float64, get(ENV, "DAYS", "5")), corner)
        @printf("%-11s %d %-7s l2 %.3e  linf %.3e  %d steps %.0f s\n", name, l, corner ? "corner" : "C grid", l2, li, n, t)
        flush(stdout)
    end
end
