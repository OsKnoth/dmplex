# Baroclinic wave with ICONNonhydrostatic on an ICON grid file, with the settings of ICON's experiment
# jabw10d (R2B5 icon_grid_0051_R02B05_G.nc, 64 levels up to 35 km by sleve_nml, dtime = 576 s with
# ndyn_substeps = 5, vwind_offctr = 0.2, exner_expol = 0.5, damp_height = 25 km, rayleigh_coeff = 0.1,
# hdiff_efdt_ratio = 20, the rest at ICON's defaults, which are those of ICONNonhydrostatic), for the
# comparison with Fortran ICON.
#
#   julia --project examples/icon_jabw.jl grid.nc [days] [dtime] [nlev] [top_height] [--gpu]
#         [--init=icon.nc] [--compare=icon.nc] [--timer]
#
# Without --init the Jablonowski–Williamson state on flat ground (ICON's `jabw` has the JW orography,
# which the model does not have). --init=icon.nc starts from the first time of an ICON output file
# (native grid, model levels, vn, w, rho, theta_v, exner), e.g. of ICON's flat DCMIP 2016 wave
# `dcmip_bw_11`; --compare=icon.nc compares the final state with the last time of an ICON output file.
# dtime is ICON's advective time step, the dynamics steps are dtime/5. Prints the minimum surface
# pressure, max |w| and the time per step once a day and writes the surface pressure (Pa) and the
# temperature of the lowest layer (K) to icon_jabw_<grid>.vtkhdf. --timer (with TimerOutputs in the
# environment) prints the time in ICON's sections (`section_timer!`) after the first advective step.
using KernelAbstractions
using Printf
using Statistics
using dmplex
using NCDatasets: NCDataset

const USE_GPU = "--gpu" in ARGS
if USE_GPU
    using CUDA
end
const USE_TIMER = "--timer" in ARGS
if USE_TIMER
    using TimerOutputs
end

const NH = dmplex.Nonhydrostatic

option(name) = (a = findfirst(startswith("--$name="), ARGS); a === nothing ? nothing : split(ARGS[a], "="; limit=2)[2])

"Surface pressure (Pa) by hydrostatic extrapolation from the lowest layer, and its temperature (K)."
function surface(m, y)
    ρ, π1 = Array(y.ρ)[:, 1], Array(y.exner)[:, 1]
    T1 = NH.P0 .* π1 .^ (NH.ICON_CVD / NH.ICON_RD) ./ (NH.ICON_RD .* ρ)
    return NH.P0 .* π1 .^ (NH.ICON_CPD / NH.ICON_RD) .* exp.(NH.ICON_GRAV * m.vgrid.z_full[1] ./ (NH.ICON_RD .* T1)), T1
end

function main(args)
    init, compare = option("init"), option("compare")
    args = filter(a -> !startswith(a, "--"), args)
    grid = args[1]
    days = length(args) >= 2 ? parse(Float64, args[2]) : 10.0
    dtime = length(args) >= 3 ? parse(Float64, args[3]) : 576.0
    nlev = length(args) >= 4 ? parse(Int, args[4]) : 64
    top = length(args) >= 5 ? parse(Float64, args[5]) : 35000.0
    nsub = 5
    dt = dtime / nsub
    backend = USE_GPU ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    dm = hilbert_order(read_grid(grid))
    v = icon_vertical_grid(nlev, top)
    m = ICONNonhydrostatic(TRiSKGrid(dm), v; beta=0.7, exner_expol=0.5, damping_height=25000.0, damping_rate=0.1,
                           hdiff_efdt_ratio=20.0, ndyn_substeps=nsub, backend)
    @printf("%d cells × %d levels (lowest %.0f m, top %.0f m) on %s, dtime = %.0f s, dynamics dt = %.1f s\n",
            num_cells(dm), nlev, v.dz[1], top, nameof(typeof(backend)), dtime, dt)
    y = zero_state(m)
    if init === nothing
        foreach(copyto!, y, initial_state(m, (λ, φ, z) -> jablonowski_williamson(λ, φ, z; radius=m.radius, omega=7.29212e-5)))
    else
        foreach(copyto!, y, read_icon_state(m, init, grid))
        println("initial state from ", init)
    end
    ys = zero_state(m)
    timer = USE_TIMER ? TimerOutput() : nothing
    section_timer!(timer)
    nsteps = round(Int, days * 86400 / dt)
    perday = round(Int, 86400 / dt)
    t0 = time()
    for n in 1:nsteps
        NH.section(() -> step!(m, y, dt, ys), m, "integrate_nh")
        n == nsub && timer !== nothing && reset_timer!(timer)   # without the compilation
        if n % perday == 0 || n == nsteps
            ps, _ = surface(m, y)
            @printf("day %5.2f  p_s min %7.2f hPa  max|w| %.3e  %.1f ms per step\n", n * dt / 86400,
                    minimum(ps) / 100, maximum(abs, Array(y.w)), 1000 * (time() - t0) / n)
            flush(stdout)
            any(isnan, ps) && break
        end
    end
    if timer !== nothing
        TimerOutputs.complement!(timer)
        show(timer; title="ICONNonhydrostatic, dynamics steps $(nsub + 1) to $nsteps")
        println()
    end
    ps, T = surface(m, y)
    name = "icon_jabw_$(splitext(basename(grid))[1]).vtkhdf"
    if compare === nothing
        write_vtkhdf(name, dm; surface_pressure=ps, temperature=T)
    else
        nt = NCDataset(ds -> length(ds["time"]), compare)
        yi = read_icon_state(m, compare, grid; step=nt)
        psi, Ti = surface(m, yi)
        d = ps .- psi
        @printf("against %s (time %d): p_s min ours %.2f hPa, ICON %.2f hPa; p_s difference rms %.2f hPa, max %.2f hPa\n",
                compare, nt, minimum(ps) / 100, minimum(psi) / 100, sqrt(mean(abs2, d)) / 100, maximum(abs, d) / 100)
        write_vtkhdf(name, dm; surface_pressure=ps, surface_pressure_icon=psi, temperature=T, temperature_icon=Ti)
    end
    println("wrote ", name)
end

main(ARGS)
