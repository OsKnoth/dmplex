# Baroclinic wave of Ullrich et al. (2014) with any dycore of dmplex (examples/dycores.jl):
#
#   julia --project examples/baroclinic_wave.jl [dycore] [resolution] [days] [dt] [levels] [options]
#   julia --project examples/baroclinic_wave.jl mpas 5 10 300 30
#   julia --project examples/baroclinic_wave.jl icon grid/icon_grid_0013_R02B04_G.nc 10 150 30 --hdiff
#   julia --project examples/baroclinic_wave.jl kite 4 10 150 30 --mesh=cubed
#   julia -t 8 --project examples/baroclinic_wave.jl poly 5 10 300 30
#   julia --project=examples/gpu examples/baroclinic_wave.jl mpas 7 10 150 30 --gpu
#
# dycore: mpas (default; MPAS-Gassmann on hexagons), icon (ICON on triangles), kite (kite elements),
# poly (polyhedral finite volumes, PolyEuler). resolution: bisections of the icosahedron (hexagon 6 ≈
# 120 km, 7 ≈ 60 km; triangles 5 ≈ 160 km, 6 ≈ 80 km), cells per panel edge for --mesh=cubed, or an
# ICON or MPAS grid file (mpas: the hexagonal dual of ICON's triangles; icon: the Delaunay triangles
# of an MPAS grid). dt in seconds; equidistant layers up to 30 km, Rayleigh damping of w above 22 km.
# Options: --mesh=hexagon|triangles|cubed, --gpu (mpas, icon, kite; environment examples/gpu),
# --out=prefix (surface fields as text, `lon lat p_s T` per column, at the days of --days=8,10),
# mpas --rk3 (Wicker–Skamarock instead of the predictor–corrector of ICON-IAP), --tridiagonal (the
# tridiagonal vertical solve for w); icon --hdiff (ICON's diffusion, hdiff_order = 5, instead of the
# fourth-order diffusion of vn); kite --velocity, --theta, --advection; poly --degree (of the
# reconstruction, 1). The old form `baroclinic_wave.jl [resolution] ... --icon` is icon, else mpas.
# Prints the minimum surface pressure and the change of mass once a day, the time per step, and
# writes p_s (Pa) and T of the lowest layer (K) to baroclinic_wave_<dycore>_<resolution>.vtkhdf
# (not for kite).
include("dycores.jl")

const USE_GPU = "--gpu" in ARGS
if USE_GPU
    using CUDA
end

function main(args)
    positional, opts = options(args)
    dycore = !isempty(positional) && positional[1] in DYCORES ? popfirst!(positional) : haskey(opts, "icon") ? "icon" : "mpas"
    resolution = length(positional) >= 1 ? something(tryparse(Int, positional[1]), positional[1]) : 5
    days = length(positional) >= 2 ? parse(Float64, positional[2]) : 10.0
    dt = length(positional) >= 3 ? parse(Float64, positional[3]) : 300.0
    levels = length(positional) >= 4 ? parse(Int, positional[4]) : 30
    backend = USE_GPU ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    mesh = get(opts, "mesh", resolution isa String ? resolution : default_mesh(dycore))
    setup = @elapsed m, dm = build_dycore(dycore, resolution; mesh, levels, backend, opts)
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z))
    @printf("%s on %s %s: %d columns × %d layers, dt = %.0f s, setup %.0f s\n", dycore, mesh, resolution,
            length(surface_pressure(m, y)), levels, dt, setup)
    flush(stdout)
    mass0 = total_mass(m, y)
    out, out_days = get(opts, "out", ""), parse.(Float64, split(get(opts, "days", "8,10"), ','))
    report(m, y, t) = begin
        @printf("day %5.2f  p_s min %7.2f hPa  max|w| %.2e m/s  mass %.1e\n", t / 86400, minimum(surface_pressure(m, y)) / 100,
                maximum(abs, vertical_velocity(m, y)), (total_mass(m, y) - mass0) / mass0)
        flush(stdout)
        isempty(out) || any(d -> abs(d * 86400 - t) < dt / 2, out_days) && write_surface(@sprintf("%s_day%02.0f.txt", out, t / 86400), m, y)
    end
    per_step = run_dycore!(m, y, dt, days; report)
    @printf("%.2f ms per step\n", 1000per_step)
    if dm !== nothing
        name = resolution isa String ? splitext(basename(resolution))[1] : resolution
        file = write_vtkhdf("baroclinic_wave_$(dycore)_$(name).vtkhdf", dm; surface_pressure=surface_pressure(m, y),
                            temperature=lowest_temperature(m, y))
        println("wrote $file")
    end
    return m, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
