# HEVI stage, the ICON-IAP predictor–corrector step and the Wicker–Skamarock RK3 step of MPAS-Gassmann.

"""
    stage!(m, y, y0, ys, τ)

One HEVI stage: `y = y0 + τ T(ys)` with the horizontal terms explicit in the
stage state `ys` and the vertical acoustic and buoyancy terms implicit,
linearized around the Exner pressure of `y0` in the work arrays (`base_exner!`).
"""
function stage!(m::MPASGassmannNonhydrostatic, y, y0, ys, τ)
    slow_tendencies!(m, ys, 1.0)
    return fast_update!(m, y, y0, ys, τ)
end

"""
    slow_tendencies!(m, ys, pressure)

The explicit tendencies of the stage state `ys`: `θ`, `π`, `K` in the work
arrays, the `vn` tendency `w.tvn` (PV flux, `-∂_n K`, vertical coupling, and
`pressure` times the pressure gradient) and the explicit `w` tendency `w.tw`.
"""
function slow_tendencies!(m::MPASGassmannNonhydrostatic, ys, pressure)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    launch!(m, diagnose_kernel!, (nc, nlev), w.θ, w.π, w.K, ys.vn, ys.ρ, ys.ρθ, m.cell_edges, m.kinetic)
    vk = m.vertex_kinetic
    if vk !== nothing
        launch!(m, vertex_kinetic_kernel!, (nv, nlev), w.Kv, ys.vn, m.vertex_edges, vk.weights)
        launch!(m, cell_from_vertex_kernel!, (nc, nlev), w.K, w.Kv, vk.cell_vertices, vk.kites)
    end
    launch!(m, vertex_pv_kernel!, (nv, nlev), w.qv, w.ζv, w.ρv, ys.vn, ys.ρ, m.vertex_edges, m.curl, m.vertex_cells,
            m.vertex_weight, m.coriolis)
    launch!(m, edge_pv_kernel!, (ne, nlev), w.qe, w.qv, w.ζv, w.ρv, m.edge_vertices, m.rhombus, m.edge_coriolis,
            m.iap_pv)
    launch!(m, edge_density_kernel!, (ne, nlev), w.ρe, w.Fs, ys.vn, ys.ρ, m.edge_cells, m.edge_weight)
    if m.vertical > 0
        # with the five-band solve ∂K_w/∂z is implicit in the w equation, in the vn equation K = K_h + K_w
        m.five_band || launch!(m, w_kinetic_kernel!, (nc, nlev), w.K, ys.w, m.dz, m.dzh)
        launch!(m, eta_kernel!, (ne, nlev + 1), w.η, ys.vn, ys.w, m.edge_cells, m.inverse_dual, m.dzh)
        launch!(m, w_tendency_vi_kernel!, (nc, nlev + 1), w.tw, w.η, ys.vn, w.ρe, w.K, m.cell_edges, m.inner, m.dzh,
                m.wlow, m.vertical == 2)
        m.five_band && launch!(m, w_kinetic_kernel!, (nc, nlev), w.K, ys.w, m.dz, m.dzh)
    else
        launch!(m, w_tendency_kernel!, (nc, nlev + 1), w.tw, ys.w, ys.vn, m.cell_edges, m.kinetic, m.edge_cells,
                m.inverse_dual, m.dz)
    end
    launch!(m, vn_tendency_kernel!, (ne, nlev), w.tvn, ys.vn, w.Fs, ys.w, ys.ρ, w.θ, w.π, w.K, w.qe, w.η,
            m.edge_cells, m.edge_weight, m.inverse_dual, m.trisk_index, m.trisk, m.dz, m.dzh, m.wlow, nlev,
            m.vertical, pressure)
    return nothing
end

"""
    fast_update!(m, y, y0, ys, τ)

`vn = vn0 + τ (w.tvn + hd)`, the horizontal fluxes with the new `vn`, `ρ` of `ys` and
`θ = w.θ`, and the vertically implicit `w`, `ρ`, `ρθ` from `y0` (`y = y0 + τ T`), with
`five_band_solve_kernel!` or `vertical_solve_kernel!`.
"""
function fast_update!(m::MPASGassmannNonhydrostatic, y, y0, ys, τ)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    launch!(m, vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn, w.hd, τ)
    launch!(m, flux_kernel!, (ne, nlev), w.F, w.Fθ, y.vn, ys.ρ, w.θ, m.edge_cells, m.edge_weight)
    launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, w.hρθ, m.cell_edges,
            m.divergence, τ)
    launch!(m, theta_interface_kernel!, nc, w.θi, w.band, w.θm, w.θ, y0.ρ, y0.ρθ, ys.ρ, ys.w, m.dz, m.dzh, m.wlow,
            τ, nlev, m.vertical_theta > 0)
    if m.five_band
        launch!(m, five_band_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.band5, y0.w, y0.ρ, w.π0, w.γ0, w.θ, w.θi, w.tw,
                w.ρexpl, w.ρθexpl, y0.ρθ, m.dz, m.dzh, m.wlow, m.rayleigh, τ, m.beta, nlev, m.vertical_theta == 2)
    else
        launch!(m, vertical_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρθ, w.π0, w.γ0, w.θ,
                w.θi, ys.ρ, w.tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, w.wmass, τ, m.beta, nlev,
                m.vertical_theta == 2, false)
    end
    return y
end

"""
    step!(m, y, dt, ys=zero_state(m))

One step of length `dt` in place, with `time_stepping = :iap` the predictor–corrector
of ICON-IAP (`iap_step!`), with `:wicker_skamarock` the RK3 of Wicker and Skamarock
(stages `dt/3`, `dt/2`, `dt`); `ys` is work space for the stage state, the base state
is kept in the work arrays (no allocation on the device).
"""
function step!(m::MPASGassmannNonhydrostatic, y, dt, ys=zero_state(m))
    y0 = m.work.y0
    foreach(copyto!, y0, y)
    hyperdiffusion!(m.work.hd, m, y0.vn)
    smagorinsky!(m.work.hd, m.work.hρθ, m, y0)
    base_exner!(m, y0)
    return m.iap_stepping ? iap_step!(m, y, y0, ys, dt) : wicker_skamarock!(m, y, y0, ys, dt)
end

"""
    iap_step!(m, y, y0, ys, dt)

The two-stage predictor–corrector of ICON-IAP (`dynamics_integration` with
`slow_modes` and `divergent_modes`), both stages from `y0` over the whole `dt`:

1. the slow tendencies `S(y0)` (PV flux, `-∂_n K`, vertical coupling, explicit
   `w` tendency), `vn* = vn0 + dt (S(y0) - c_p θ_e(θ0) ∂_n π_m)`, the fluxes with
   `vn*` and the vertically implicit `w*`, `ρ*`, `ρθ*` (the predictor `ys`);
2. the same with the mean `(S(y0) + S(ys))/2` and `θ = (θ0 + θ*)/2` in the pressure
   gradient and the `ρθ` flux, `ρ` of the predictor in the mass flux, into `y`.

The horizontal pressure gradient takes the Exner pressure off-centred in time,
`π_m = (R/c_p - 1/2) π^{n-1} + (3/2 - R/c_p) π^n` (`π^{n-1} = π^n` in the first step).
"""
function iap_step!(m::MPASGassmannNonhydrostatic, y, y0, ys, dt)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne = num_cells(m.grid), num_edges(m.grid)
    w.started[] || (copyto!(w.πold, w.π0); w.started[] = true)
    launch!(m, exner_mean_kernel!, (nc, nlev), w.πm, w.πold, w.π0)
    # predictor
    slow_tendencies!(m, y0, 0.0)
    copyto!(w.tvn1, w.tvn); copyto!(w.tw1, w.tw); copyto!(w.θs, w.θ)
    launch!(m, pressure_gradient_kernel!, (ne, nlev), w.tvn, w.θ, w.πm, m.edge_cells, m.edge_weight, m.inverse_dual)
    fast_update!(m, ys, y0, y0, dt)
    # corrector
    slow_tendencies!(m, ys, 0.0)
    launch!(m, mean_kernel!, (ne, nlev), w.tvn, w.tvn1)
    launch!(m, mean_kernel!, (nc, nlev + 1), w.tw, w.tw1)
    launch!(m, mean_kernel!, (nc, nlev), w.θ, w.θs)
    launch!(m, pressure_gradient_kernel!, (ne, nlev), w.tvn, w.θ, w.πm, m.edge_cells, m.edge_weight, m.inverse_dual)
    return fast_update!(m, y, y0, ys, dt)
end

"Exner pressure of the base state `y0` and its derivative by `ρθ` into the work arrays, for `stage!`."
function base_exner!(m::MPASGassmannNonhydrostatic, y0)
    launch!(m, base_exner_kernel!, size(y0.ρθ), m.work.π0, m.work.γ0, y0.ρθ)
    return nothing
end
