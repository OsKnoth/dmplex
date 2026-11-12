# Two-time-level predictor–corrector step of ICON.

"""
    step!(m::ICONNonhydrostatic, y, dt, ys=zero_state(m))

One predictor–corrector step of ICON (see `ICONNonhydrostatic`) in place;
`ys` holds the predictor state. With `section_timer!` timed in ICON's sections:
nh_solve with veltend (velocity tendencies), cellcomp (θ, π, extrapolated π),
vnupd (ρ_e, θ_e, pressure gradient, vn update, divergence damping), edgecomp
(averaged vn, fluxes), vimpl (flux divergence, column solve); nh_hdiff.
"""
function step!(m::ICONNonhydrostatic, y, dt, ys=zero_state(m))
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    y0 = w.y0
    section(m, "nh_solve") do
        foreach(copyto!, y0, y)
        copyto!(w.π0, y0.exner)
        recompute = m.itime_scheme == 6 || w.first[]
        w.first[] && (copyto!(w.πprev, w.π0); w.first[] = false)
        # predictor: velocity tendencies of n (itime_scheme = 4: of the last corrector, but that of vn
        # recomputed in the first step after the diffusion, ICON `l_recompute`), ρ_e, θ_e and the pressure
        # gradient of the extrapolated π
        if recompute || w.substep[] == 0
            section(() -> velocity_tendencies!(w.tvn_now, w.tw_now, m, y0; vn_only=!recompute), m, "veltend")
        end
        section(m, "cellcomp") do
            if !recompute && w.substep[] != 0
                launch!(m, icon_thermo_kernel!, (nc, nlev), w.θ, w.π, y0.ρ, y0.ρθ)
                launch!(m, icon_edge_kinetic_kernel!, (ne, nlev), w.vt, w.Ke, y0.vn, m.rbf_edges, m.rbf)
            end
            f = m.vertical.exfac   # exner_expol, reduced over steep slopes
            @. w.π = (1 + f) * w.π0 - f * w.πprev
        end
        section(m, "vnupd") do
            m.hyperdiffusion == 0 ? fill!(w.hd, 0.0) : icon_hyperdiffusion!(w.hd, m, y0.vn)
            g, v = w.grad, m.vertical
            # deviations from the reference state for the Miura reconstruction (ICON z_rth_pr)
            @. w.ρpr = y0.ρ - v.ρref_c
            @. w.θpr = w.θ - v.θref_c
            launch!(m, icon_cell_gradient_kernel!, (nc, nlev), g..., w.ρpr, w.θpr, m.cell_edges, m.edge_cells,
                    m.edge_height, m.green_gauss...)
            m.orography && launch!(m, exner_deviation_kernel!, (nc, nlev), w.πp, w.dπdz, w.π, m.vertical.πref,
                                   m.wlow, m.vertical.wgtfacq, m.dz, nlev)
            x = m.metric
            taylor = m.orography && x.pgrad == 3
            taylor && launch!(m, icon_exner_curvature_kernel!, (nc, nlev), w.c2, w.dθdz, w.θ, v.θref, x.curv_A, x.curv_B,
                              m.wlow, v.wgtfacq, m.dz, nlev)
            launch!(m, icon_miura_kernel!, (ne, nlev), w.ρe, w.θe, w.pg, y0.vn, w.vt, w.ρpr, w.θpr,
                    m.orography ? w.πp : w.π, g..., m.edge_cells, m.edge_vectors, m.inverse_dual, dt / (2 * m.radius),
                    w.dπdz, x.ddxn, m.edge_height, x.zlev, x.zwt, m.orography ? x.pgrad : 0, v.ρref_e, v.θref_e,
                    w.c2, x.ntaylor)
            p = x.gradp_ex
            taylor && !isempty(p.e) && launch!(m, icon_gradp_extrapolation_kernel!, (length(p.e),), w.pg, w.θe, w.θ,
                                               w.dθdz, p.e, p.k, p.dist, x.zlev, x.zwt, m.edge_cells, m.inverse_dual)
            copyto!(w.πprev, w.π0)
            launch!(m, icon_vn_update_kernel!, (ne, nlev), ys.vn, y0.vn, w.tvn_now, w.tvn_now, w.pg, w.hd, dt, 1.0, 0.0)
        end
        fast_waves!(m, ys, ys.exner, y0, y0, w.tw_now, dt, false)
        # corrector: velocity tendencies of the predictor state, weighted, and the divergence damping of it
        section(() -> velocity_tendencies!(w.tvn_new, w.tw_new, m, ys), m, "veltend")
        a, b = m.weights
        section(m, "vnupd") do
            launch!(m, icon_vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn_now, w.tvn_new, w.pg, w.hd, dt, a, b)
            divergence_damping!(y.vn, m, ys)
        end
        @. w.tw = a * w.tw_now + b * w.tw_new
        fast_waves!(m, y, y.exner, y0, ys, w.tw, dt, true)
        # the corrector tendencies are those of n in the next predictor (ICON swaps nnow and nnew)
        if m.itime_scheme == 4
            copyto!(w.tvn_now, w.tvn_new)
            copyto!(w.tw_now, w.tw_new)
        end
    end
    # mo_nh_diffusion after ndyn_substeps dynamics steps, over the advective time step
    w.substep[] += 1
    if w.substep[] == m.ndyn_substeps
        m.hdiff_order == 5 && section(() -> icon_diffusion!(m, y, m.ndyn_substeps * dt), m, "nh_hdiff")
        w.substep[] = 0
    end
    return y
end
