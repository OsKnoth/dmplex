# Parts shared by all nonhydrostatic dycores: physical constants, the abstract C-grid model type and
# the kernel launcher with the blocks of points × levels.

"""
Nonhydrostatic models on an extruded C grid (`MPASGassmannNonhydrostatic`,
`ICONNonhydrostatic`): state `(vn, w, ρ, ρθ)` with `vn` at the edges × layers,
`w` at the cells × interfaces, `ρ`, `ρθ` at the cells × layers, and fields
`grid`, `vgrid`, `cell_areas`, `dz`, `vertical_theta`, `backend`.
"""
abstract type CGridNonhydrostatic end

const Rd = 287.0
const Cpd = 1004.0
const Cvd = Cpd - Rd
const P0 = 1.0e5
const GRAV = 9.80616

# ICON's constants (mo_physical_constants), used by ICONNonhydrostatic; the analytic test cases keep the ones
# above, as their DCMIP/JW formulas do in ICON
const ICON_RD = 287.04
const ICON_CPD = 1004.64
const ICON_CVD = ICON_CPD - ICON_RD
const ICON_GRAV = 9.80665

# Work items over (points, levels) run in blocks of 32 points × 8 levels: the neighbour tables and the
# lines of the gathered neighbours are shared by eight levels in the cache (ICON's nproma blocks).
const BLOCK = (32, 8)

function launch!(m, kernel, ndrange, args...)   # a model with a `backend`
    k = ndrange isa Tuple{Integer, Integer} ? kernel(m.backend, BLOCK) : kernel(m.backend)
    k(args...; ndrange)
    KernelAbstractions.synchronize(m.backend)
    return nothing
end

# The timer of the sections of a step (`section_timer!`), `nothing` when they are not timed
const SECTION_TIMER = Ref{Any}(nothing)

"""
    section_timer!(t)

Times the sections of the steps of the models (ICONNonhydrostatic: ICON's timers
nh_solve, veltend, cellcomp, vnupd, edgecomp, vimpl, nh_hdiff) in the
`TimerOutput` `t` (with `using TimerOutputs`; the backend synchronized at the
start and end of each section), or not with `nothing` (the default).
"""
section_timer!(t) = (SECTION_TIMER[] = t; nothing)

"`f()` as the section `name` of model `m`; timed by the extension for TimerOutputs."
section(f, m, name) = section(f, SECTION_TIMER[], m, name)
section(f, ::Nothing, m, name) = f()
