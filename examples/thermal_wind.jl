# Steady rotating wind in thermal-wind balance (`thermal_wind`) with any dycore of dmplex
# (examples/dycores.jl): a test of how well a model keeps a balanced, stratified, sheared flow at rest.
#
#   julia --project examples/thermal_wind.jl [dycore] [resolution] [days] [dt] [options]
#   julia --project examples/thermal_wind.jl mpas 4 5 300
#   julia --project examples/thermal_wind.jl kite 3 5 150 --mesh=cubed --velocity=dual_radau
#   julia -t 8 --project examples/thermal_wind.jl poly 4 5 300
#
# dycore: mpas (default, also `trisk`), icon, kite, poly; options as in baroclinic_wave.jl. The old kite
# form `kite [resolution] [days] [dt] [velocity] [theta] [mesh] [advection]` still works. 30
# equidistant layers up to 30 km, Rayleigh damping of w above 22 km. Prints twice a day the largest
# |w|, the largest change of the temperature of the lowest layer and of the surface pressure.
include("dycores.jl")

function main(args)
    positional, opts = options(args)
    dycore = isempty(positional) ? "mpas" : popfirst!(positional)
    dycore == "trisk" && (dycore = "mpas")
    resolution = length(positional) >= 1 ? parse(Int, positional[1]) : (dycore == "kite" ? 3 : 4)
    days = length(positional) >= 2 ? parse(Float64, positional[2]) : 5.0
    dt = length(positional) >= 3 ? parse(Float64, positional[3]) : (dycore == "kite" ? 150.0 : 300.0)
    for (i, key) in enumerate(("velocity", "theta", "mesh", "advection"))     # the old kite arguments
        dycore == "kite" && length(positional) >= 3 + i && (opts[key] = positional[3 + i])
    end
    mesh = get(opts, "mesh", default_mesh(dycore))
    m, _ = build_dycore(dycore, resolution; mesh, opts)
    y = initial_state(m, (λ, φ, z) -> thermal_wind(λ, φ, z))
    T0, ps0 = lowest_temperature(m, y), surface_pressure(m, y)
    @printf("%s, %s %d, dt = %.0f s\n", dycore, mesh, resolution, dt)
    run_dycore!(m, y, dt, days; every=43200.0, report=(m, y, t) ->
        @printf("day %4.1f  max|w| %.2e m/s  max|ΔT| %.2e K  max|Δp_s| %.2e hPa\n", t / 86400,
                maximum(abs, vertical_velocity(m, y)), maximum(abs, lowest_temperature(m, y) .- T0),
                maximum(abs, surface_pressure(m, y) .- ps0) / 100))
    return m, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
