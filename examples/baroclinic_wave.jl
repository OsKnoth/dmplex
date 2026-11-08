# Baroclinic wave of Ullrich et al. (2014) with the nonhydrostatic TRiSK model on
# the extruded hexagonal mesh (ICON-IAP form: PV usage, momentum advection, TRiSK
# weights, Smagorinsky diffusion, upwind vertical θ), or with --icon the ICON model
# on the extruded triangle mesh (predictor–corrector, Miura fluxes, divergence
# damping, fourth-order diffusion of vn), on one CPU or one GPU.
#
#   julia --project examples/baroclinic_wave.jl [resolution] [days] [dt] [levels] [--icon] [--hdiff] [--rk3]
#   julia --project examples/baroclinic_wave.jl grid/icon_grid_0013_R02B04_G.nc 10 150 30 --icon --hdiff
#   julia --project=examples/gpu examples/baroclinic_wave.jl 7 10 150 30 --gpu
#
# resolution: refinement level of the hexagonal mesh (6 ≈ 120 km, 7 ≈ 60 km) or of
# the triangle mesh (20·4ⁿ triangles: 5 ≈ 160 km, 6 ≈ 80 km, 7 ≈ 40 km, for ICON with
# dt ≈ 150, 75, 37.5 s), or an ICON or MPAS grid file (ICON's triangles for --icon,
# their hexagonal dual for TRiSK; the Delaunay triangles of an MPAS grid for --icon);
# --rk3 steps MPAS-Gassmann with Wicker–Skamarock RK3 instead of the predictor–corrector of
# ICON-IAP; --hdiff uses ICON's diffusion (hdiff_order = 5) instead of the fourth-order diffusion
# of vn (on the meshes of create_spherical_mesh it is not stable); dt in seconds; 30 equidistant layers up to 30 km, Rayleigh
# damping of w above 22 km.
# --gpu runs on the CUDA GPU (environment examples/gpu). Prints the minimum surface
# pressure once a day and the time per step; writes p_s (Pa) and T of the lowest
# layer (K) at the cells to baroclinic_wave_hexagon_<resolution>.vtkhdf.
using KernelAbstractions
using NCDatasets: NCDataset
using Printf
using dmplex

const USE_GPU = "--gpu" in ARGS
const USE_ICON = "--icon" in ARGS
const USE_HDIFF = "--hdiff" in ARGS
const USE_RK3 = "--rk3" in ARGS
if USE_GPU
    using CUDA
end

const NH = dmplex.Nonhydrostatic

"Surface pressure (Pa) and temperature of the lowest layer (K) at the cells, on the host."
function surface(m, y)
    ρ, ρθ = Array(y.ρ), Array(y.ρθ)
    π1 = (NH.Rd .* ρθ[:, 1] ./ NH.P0) .^ (NH.Rd / NH.Cvd)
    T1 = π1 .* ρθ[:, 1] ./ ρ[:, 1]
    ps = NH.P0 .* π1 .^ (NH.Cpd / NH.Rd) .* exp.(NH.GRAV * m.vgrid.z_full[1] ./ (NH.Rd .* T1))
    return ps, T1
end

function main(args)
    args = filter(a -> !startswith(a, "--"), args)
    resolution = length(args) >= 1 ? something(tryparse(Int, args[1]), args[1]) : 5
    days = length(args) >= 2 ? parse(Float64, args[2]) : 10.0
    dt = length(args) >= 3 ? parse(Float64, args[3]) : 300.0
    levels = length(args) >= 4 ? parse(Int, args[4]) : 30

    backend = USE_GPU ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    # neighbours close in memory
    dm = hilbert_order(if resolution isa String
        icon_file = NCDataset(ds -> haskey(ds, "vertex_of_cell"), resolution)
        read_grid(resolution; dual=icon_file ? !USE_ICON : USE_ICON)
    else
        USE_ICON ? create_spherical_mesh(resolution) : create_hexagon_mesh(resolution)
    end)
    v = VerticalGrid(levels, 30000.0)
    m = if USE_ICON
        # ICON's diffusion, or a fourth-order diffusion of vn scaled as ICON's (∝ A²/Δt ∝ Δx³),
        # 5e14 m⁴/s at 79 km (R2B5)
        Δx = sqrt(4π / num_cells(dm)) * 6.37122e6
        ICONNonhydrostatic(TRiSKGrid(dm), v; damping_height=22000.0, damping_rate=0.05,
                           hdiff_order=USE_HDIFF ? 5 : -1, hyperdiffusion=USE_HDIFF ? 0.0 : 5e14 * (Δx / 79e3)^3,
                           backend)
    else
        MPASGassmannNonhydrostatic(TRiSKGrid(dm), v; damping_height=22000.0, damping_rate=0.05,
                                   time_stepping=USE_RK3 ? :wicker_skamarock : :iap, backend)
    end
    @printf("%d cells (≈ %.0f km) × %d layers on %s, dt = %.0f s\n", num_cells(dm),
            sqrt(4π / num_cells(dm)) * 6.37122e6 / 1000, levels, nameof(typeof(backend)), dt)
    y = initial_state(m, (λ, φ, z) -> baroclinic_wave(λ, φ, z))
    ys = zero_state(m)
    mass0 = NH.total_mass(m, y)
    step!(m, map(copy, y), dt, ys)   # compile the kernels before timing
    per_day, nsteps = round(Int, 86400 / dt), round(Int, days * 86400 / dt)
    elapsed = 0.0
    for n in 1:nsteps
        elapsed += @elapsed step!(m, y, dt, ys)
        if n % per_day == 0 || n == nsteps
            ps, _ = surface(m, y)
            @printf("day %5.2f  p_s min %7.2f hPa  mass %.1e\n", n * dt / 86400, minimum(ps) / 100,
                    (NH.total_mass(m, y) - mass0) / mass0)
        end
    end
    @printf("%.1f s in %d time steps, %.2f ms per step\n", elapsed, nsteps, 1000elapsed / nsteps)
    ps, T = surface(m, y)
    filename = write_vtkhdf("baroclinic_wave_$(USE_ICON ? "triangles" : "hexagon")_$(resolution isa String ? splitext(basename(resolution))[1] : resolution).vtkhdf", dm; surface_pressure=ps, temperature=T)
    println("wrote $filename")
    return m, y
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
