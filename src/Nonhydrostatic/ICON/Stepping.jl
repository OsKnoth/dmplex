# Two-time-level predictor–corrector step of ICON.

"""
    step!(m::ICONNonhydrostatic, y, dt, ys=zero_state(m))

One predictor–corrector step of ICON (see `ICONNonhydrostatic`) in place;
`ys` holds the predictor state.
"""
function step!(m::ICONNonhydrostatic, y, dt, ys=zero_state(m))
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    y0 = w.y0
    foreach(copyto!, y0, y)
    copyto!(w.π0, y0.exner)
    recompute = m.itime_scheme == 6 || w.first[]
    w.first[] && (copyto!(w.πprev, w.π0); w.first[] = false)
    m.hyperdiffusion == 0 ? fill!(w.hd, 0.0) : icon_hyperdiffusion!(w.hd, m, y0.vn)
    # predictor: velocity tendencies of n (itime_scheme = 4: of the last corrector, but that of vn
    # recomputed in the first step after the diffusion, ICON `l_recompute`), ρ_e, θ_e and the pressure
    # gradient of the extrapolated π
    if recompute
        velocity_tendencies!(w.tvn_now, w.tw_now, m, y0)
    elseif w.substep[] == 0
        velocity_tendencies!(w.tvn_now, w.tw_now, m, y0; vn_only=true)
    else
        launch!(m, icon_thermo_kernel!, (nc, nlev), w.θ, w.π, y0.ρ, y0.ρθ)
        launch!(m, icon_edge_kinetic_kernel!, (ne, nlev), w.vt, w.Ke, y0.vn, m.rbf_edges, m.rbf)
    end
    @. w.π = (1 + m.exner_expol) * w.π0 - m.exner_expol * w.πprev
    g = w.grad
    launch!(m, icon_cell_gradient_kernel!, (nc, nlev), g..., y0.ρ, w.θ, m.cell_edges, m.edge_cells, m.edge_height,
            m.green_gauss...)
    launch!(m, icon_miura_kernel!, (ne, nlev), w.ρe, w.θe, w.pg, y0.vn, w.vt, y0.ρ, w.θ, w.π, g..., m.edge_cells,
            m.edge_vectors, m.inverse_dual, dt / (2 * m.radius))
    copyto!(w.πprev, w.π0)
    launch!(m, icon_vn_update_kernel!, (ne, nlev), ys.vn, y0.vn, w.tvn_now, w.tvn_now, w.pg, w.hd, dt, 1.0, 0.0)
    fast_waves!(m, ys, ys.exner, y0, y0, w.tw_now, dt, false)
    # corrector: velocity tendencies of the predictor state, weighted, and the divergence damping of it
    velocity_tendencies!(w.tvn_new, w.tw_new, m, ys)
    a, b = m.weights
    launch!(m, icon_vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn_now, w.tvn_new, w.pg, w.hd, dt, a, b)
    divergence_damping!(y.vn, m, ys)
    @. w.tw = a * w.tw_now + b * w.tw_new
    fast_waves!(m, y, y.exner, y0, ys, w.tw, dt, true)
    # the corrector tendencies are those of n in the next predictor (ICON swaps nnow and nnew)
    if m.itime_scheme == 4
        copyto!(w.tvn_now, w.tvn_new)
        copyto!(w.tw_now, w.tw_new)
    end
    # mo_nh_diffusion after ndyn_substeps dynamics steps, over the advective time step
    w.substep[] += 1
    if w.substep[] == m.ndyn_substeps
        m.hdiff_order == 5 && icon_diffusion!(m, y, m.ndyn_substeps * dt)
        w.substep[] = 0
    end
    return y
end
