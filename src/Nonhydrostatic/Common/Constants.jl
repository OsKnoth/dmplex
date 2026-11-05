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

# Work items over (points, levels) run in blocks of 32 points × 8 levels: the neighbour tables and the
# lines of the gathered neighbours are shared by eight levels in the cache (ICON's nproma blocks).
const BLOCK = (32, 8)

function launch!(m, kernel, ndrange, args...)   # a model with a `backend`
    k = ndrange isa Tuple{Integer, Integer} ? kernel(m.backend, BLOCK) : kernel(m.backend)
    k(args...; ndrange)
    KernelAbstractions.synchronize(m.backend)
    return nothing
end
