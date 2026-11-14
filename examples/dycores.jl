# The dycores of dmplex behind one interface, for the test-case examples (baroclinic_wave.jl,
# thermal_wind.jl): a test case is a function `profile(λ, φ, z) -> (T, p, u, v)` and runs with any of
#
#   mpas   MPASGassmannNonhydrostatic, TRiSK on hexagons (or the Voronoi dual of a grid file)
#   icon   ICONNonhydrostatic, ICON's C grid on triangles
#   kite   KiteNonhydrostatic, compatible kite elements RT1 × Q2 on hexagons, cubed sphere or triangles
#   poly   PolyEuler, polyhedral finite volumes on prisms over hexagons, triangles or the cubed sphere
#
# through the common interface `initial_state(m, profile)`, `zero_state(m)`, `step!(m, y, dt, ys)`,
# `total_mass(m, y)`, `surface_pressure(m, y)`, `lowest_temperature(m, y)` and `vertical_velocity(m, y)`.
# Include it and call `build_dycore`; `options(args)` splits the command line into positional
# arguments and `--key=value` / `--flag` options.
using KernelAbstractions
using LinearAlgebra
using NCDatasets: NCDataset
using Printf
using dmplex

const NH = dmplex.Nonhydrostatic
const DYCORES = ("mpas", "icon", "kite", "poly")
const EARTH_RADIUS = 6.37122e6

"Positional arguments and a Dict of the `--key=value` (value `\"true\"` for `--flag`) options."
function options(args)
    positional = filter(a -> !startswith(a, "--"), args)
    opts = Dict{String, String}()
    for a in filter(startswith("--"), args)
        k, v = occursin('=', a) ? split(a[3:end], '='; limit=2) : (a[3:end], "true")
        opts[k] = v
    end
    return positional, opts
end

"The 2D mesh `mesh` (hexagon, triangles, cubed, or a grid file) at `resolution`."
function base_mesh(mesh, resolution; dual=false)
    if isfile(string(mesh))
        icon_file = NCDataset(ds -> haskey(ds, "vertex_of_cell"), mesh)
        return read_grid(mesh; dual=icon_file ? dual : !dual)
    end
    mesh in ("hexagon", "hexagons") && return create_hexagon_mesh(resolution)
    mesh in ("triangle", "triangles", "icosahedron") && return create_spherical_mesh(resolution)
    mesh == "cubed" && return create_cubed_sphere_mesh(resolution)
    throw(ArgumentError("unknown mesh $mesh"))
end

default_mesh(dycore) = dycore == "icon" ? "triangles" : "hexagon"

"""
    build_dycore(dycore, resolution; mesh=default_mesh(dycore), levels=30, top=30000,
                 damping_height=22000, damping_rate=0.05, backend=CPU(), opts=Dict()) -> (m, dm)

The model `dycore` on `mesh` at `resolution` (bisections of the icosahedron, or cells per panel
edge on the cubed sphere) with `levels` equidistant layers up to `top` and Rayleigh damping of
`w` above `damping_height`; `dm` is the 2D mesh whose cells carry the surface fields (`nothing`
for kite, whose columns are the height dofs). Dycore options in `opts`: mpas `rk3`,
`tridiagonal`; icon `hdiff`; kite `velocity`, `theta`, `advection`; poly `degree`,
`upwind` (weight of the upwinding of the momentum transport, 0: centred), `nu4` (hyperviscosity,
m⁴/s, 3e15 at 223 km, ∝ Δx³).
"""
function build_dycore(dycore, resolution; mesh=default_mesh(dycore), levels::Integer=30, top::Real=30000.0,
                      damping_height::Real=22000.0, damping_rate::Real=0.05,
                      backend=KernelAbstractions.CPU(), opts=Dict{String, String}())
    dycore in DYCORES || throw(ArgumentError("dycore must be one of $(join(DYCORES, ", "))"))
    v = VerticalGrid(levels, top)
    if dycore == "mpas"
        dm = hilbert_order(base_mesh(mesh, resolution; dual=true))
        m = MPASGassmannNonhydrostatic(TRiSKGrid(dm), v; damping_height, damping_rate,
                                       time_stepping=haskey(opts, "rk3") ? :wicker_skamarock : :iap,
                                       vertical_solver=haskey(opts, "tridiagonal") ? :tridiagonal : :five_band, backend)
        return m, dm
    elseif dycore == "icon"
        dm = hilbert_order(base_mesh(mesh, resolution))
        # ICON's diffusion, or a fourth-order diffusion of vn scaled as ICON's (∝ Δx³), 5e14 m⁴/s at 79 km
        Δx = sqrt(4π / num_cells(dm)) * EARTH_RADIUS
        hdiff = haskey(opts, "hdiff")
        m = ICONNonhydrostatic(TRiSKGrid(dm), v; damping_height, damping_rate, hdiff_order=hdiff ? 5 : -1,
                               hyperdiffusion=hdiff ? 0.0 : 5e14 * (Δx / 79e3)^3, backend)
        return m, dm
    elseif dycore == "kite"
        dm = base_mesh(mesh, resolution)
        velocity = Symbol(get(opts, "velocity", "rt"))
        m = KiteNonhydrostatic(dm, v, 1; height_order=velocity == :rt ? 2 : 1, velocity,
                               horizontal_theta=Symbol(get(opts, "theta", "dg_upwind")),
                               horizontal_advection=Symbol(get(opts, "advection", "dg")),
                               damping_height, damping_rate, backend)
        return m, nothing
    else
        dm = base_mesh(mesh, resolution)
        cx = extruded_complex(ExtrudedPlex(dm, levels), v.z_half; radius=EARTH_RADIUS)
        # centred momentum transport with a hyperviscosity ∝ Δx³, 3e15 m⁴/s at 223 km (hexagon 5)
        Δx = sqrt(4π / num_cells(dm)) * EARTH_RADIUS
        nu4 = haskey(opts, "nu4") ? parse(Float64, opts["nu4"]) : 3e15 * (Δx / 223e3)^3
        m = PolyEuler(cx, v.z_half; radius=EARTH_RADIUS, damping_height, damping_rate,
                      degree=parse(Int, get(opts, "degree", "1")),
                      momentum_upwind=parse(Float64, get(opts, "upwind", "0")), hyperviscosity=nu4)
        return m, dm
    end
end

"Longitude and latitude (radians) of the points of `surface_pressure(m, y)`."
function surface_coordinates(m)
    lonlat(x) = (atan(x[2], x[1]), asin(clamp(x[3] / norm(x), -1, 1)))
    xs = m isa PolyEuler ? [m.geometry.cell_centroid[(i - 1) * m.layers + 1] for i in 1:(length(m.geometry.volume) ÷ m.layers)] :
         m isa KiteNonhydrostatic ? NH.column_points(m) : m.grid.centers
    return lonlat.(xs)
end

"Write `lon lat p_s T_lowest` (degrees, Pa, K) of every column to `path`."
function write_surface(path, m, y)
    ps, T = surface_pressure(m, y), lowest_temperature(m, y)
    open(path, "w") do io
        for ((λ, φ), p, t) in zip(surface_coordinates(m), ps, T)
            @printf(io, "%.6f %.6f %.3f %.4f\n", rad2deg(λ), rad2deg(φ), p, t)
        end
    end
    return path
end

"""
    run_dycore!(m, y, dt, days; every=86400, report=(m, y, t) -> nothing) -> wall time per step

`days` of `step!`, calling `report(m, y, t)` every `every` seconds (and at the end); the first step
is timed apart (compilation).
"""
function run_dycore!(m, y, dt, days; every=86400.0, report=(m, y, t) -> nothing)
    ys = zero_state(m)
    nsteps, per = round(Int, days * 86400 / dt), max(1, round(Int, every / dt))
    elapsed = 0.0
    for n in 1:nsteps
        t = @elapsed step!(m, y, dt, ys)
        n > 1 && (elapsed += t)
        (n % per == 0 || n == nsteps) && report(m, y, n * dt)
    end
    return elapsed / max(1, nsteps - 1)
end
