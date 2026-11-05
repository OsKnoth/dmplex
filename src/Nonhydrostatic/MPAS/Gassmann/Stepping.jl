# HEVI stage and Wicker–Skamarock RK3 step of MPAS-Gassmann.

"""
    stage!(m, y, y0, ys, τ)

One HEVI stage: `y = y0 + τ T(ys)` with the horizontal terms explicit in the
stage state `ys` and the vertical acoustic and buoyancy terms implicit,
linearized around the Exner pressure of `y0` in the work arrays (`base_exner!`).
"""
function stage!(m::MPASGassmannNonhydrostatic, y, y0, ys, τ)
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
        launch!(m, w_kinetic_kernel!, (nc, nlev), w.K, ys.w, m.dz, m.dzh)
        launch!(m, eta_kernel!, (ne, nlev + 1), w.η, ys.vn, ys.w, m.edge_cells, m.inverse_dual, m.dzh)
        launch!(m, w_tendency_vi_kernel!, (nc, nlev + 1), w.tw, w.η, ys.vn, w.ρe, w.K, m.cell_edges, m.inner, m.dzh,
                m.wlow, m.vertical == 2)
    else
        launch!(m, w_tendency_kernel!, (nc, nlev + 1), w.tw, ys.w, ys.vn, m.cell_edges, m.kinetic, m.edge_cells,
                m.inverse_dual, m.dz)
    end
    launch!(m, vn_tendency_kernel!, (ne, nlev), w.tvn, ys.vn, w.Fs, ys.w, ys.ρ, w.θ, w.π, w.K, w.qe, w.η,
            m.edge_cells, m.edge_weight, m.inverse_dual, m.trisk_index, m.trisk, m.dz, m.dzh, m.wlow, nlev,
            m.vertical)
    launch!(m, vn_update_kernel!, (ne, nlev), y.vn, y0.vn, w.tvn, w.hd, τ)
    launch!(m, flux_kernel!, (ne, nlev), w.F, w.Fθ, y.vn, ys.ρ, w.θ, m.edge_cells, m.edge_weight)
    launch!(m, divergence_kernel!, (nc, nlev), w.ρexpl, w.ρθexpl, y0.ρ, y0.ρθ, w.F, w.Fθ, w.hρθ, m.cell_edges,
            m.divergence, τ)
    launch!(m, theta_interface_kernel!, nc, w.θi, w.band, w.θm, w.θ, y0.ρ, y0.ρθ, ys.ρ, ys.w, m.dz, m.dzh, m.wlow,
            τ, nlev, m.vertical_theta > 0)
    launch!(m, vertical_solve_kernel!, nc, y.w, y.ρ, y.ρθ, w.cprime, w.dprime, y0.w, y0.ρθ, w.π0, w.γ0, w.θ, w.θi,
            ys.ρ, w.tw, w.ρexpl, w.ρθexpl, m.dz, m.dzh, m.wlow, m.rayleigh, w.wmass, τ, m.beta, nlev,
            m.vertical_theta == 2, false)
    return y
end

"""
    step!(m, y, dt, ys=zero_state(m))

One Wicker–Skamarock RK3 step of length `dt` (stages `dt/3`, `dt/2`, `dt`),
in place; `ys` is work space for the stage state, the base state is kept in
the work arrays (no allocation on the device).
"""
function step!(m::MPASGassmannNonhydrostatic, y, dt, ys=zero_state(m))
    y0 = m.work.y0
    foreach(copyto!, y0, y)
    hyperdiffusion!(m.work.hd, m, y0.vn)
    smagorinsky!(m.work.hd, m.work.hρθ, m, y0)
    base_exner!(m, y0)
    return wicker_skamarock!(m, y, y0, ys, dt)
end

"Exner pressure of the base state `y0` and its derivative by `ρθ` into the work arrays, for `stage!`."
function base_exner!(m::MPASGassmannNonhydrostatic, y0)
    launch!(m, base_exner_kernel!, size(y0.ρθ), m.work.π0, m.work.γ0, y0.ρθ)
    return nothing
end
