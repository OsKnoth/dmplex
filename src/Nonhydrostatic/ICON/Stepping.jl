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
    launch!(m, base_exner_kernel!, (nc, nlev), w.π0, w.γ0, y0.ρθ)
    w.first[] && (copyto!(w.πprev, w.π0); w.first[] = false)
    m.hyperdiffusion == 0 ? fill!(w.hd, 0.0) : icon_hyperdiffusion!(w.hd, m, y0.vn)
    # predictor: velocity tendencies of n, ρ_e, θ_e and the pressure gradient of the extrapolated π
    velocity_tendencies!(w.tvn_now, w.tw_now, m, y0)
    @. w.π = (1 + m.exner_expol) * w.π0 - m.exner_expol * w.πprev
    g = w.grad
    launch!(m, icon_cell_gradient_kernel!, (nc, nlev), g..., y0.ρ, w.θ, m.cell_edges, m.edge_cells, m.edge_height,
            m.green_gauss...)
    launch!(m, icon_miura_kernel!, (ne, nlev), w.ρe, w.θe, w.pg, y0.vn, w.vt, y0.ρ, w.θ, w.π, g..., m.edge_cells,
            m.edge_vectors, m.inverse_dual, dt / (2 * m.radius))
    copyto!(w.πprev, w.π0)
    launch!(m, icon_vn_update_kernel!, (ne, nlev), ys.vn, y0.vn, w.tvn_now, w.tvn_now, w.pg, w.hd, dt, 1.0, 0.0)
    fast_waves!(m, ys, y0, y0, w.tw_now, dt)
    # corrector: velocity tendencies of the predictor state, weighted, and the divergence damping of vn*
    velocity_tendencies!(w.tvn_new, w.tw_new, m, ys)
    a, b = m.weights
    launch!(m, icon_vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn_now, w.tvn_new, w.pg, w.hd, dt, a, b)
    if m.divdamp > 0
        launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, ys.vn, m.cell_edges, m.divergence)
        launch!(m, icon_edge_gradient_kernel!, (ne, nlev), w.G, w.δ, m.edge_cells, m.inverse_dual, 1.0)
        launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, w.G, m.cell_edges, m.divergence)
        launch!(m, icon_edge_gradient_kernel!, (ne, nlev), w.G, w.δ, m.edge_cells, m.inverse_dual, -m.divdamp)
        y.vn .+= w.G
    end
    @. w.tw = a * w.tw_now + b * w.tw_new
    fast_waves!(m, y, y0, ys, w.tw, dt)
    return y
end
