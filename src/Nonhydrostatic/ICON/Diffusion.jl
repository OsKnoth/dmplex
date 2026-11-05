# Horizontal diffusion of ICON: the fourth-order hyperdiffusion of vn (in place of `mo_nh_diffusion`).

"Hyperdiffusion `-ν₄ L(L(vn))` with the C-grid vector Laplacian `∇δ - k × ∇ζ`."
function icon_hyperdiffusion!(hd, m::ICONNonhydrostatic, vn)
    w, nlev = m.work, num_layers(m.vgrid)
    nc, ne, nv = num_cells(m.grid), num_edges(m.grid), num_vertices(m.grid)
    for (out, x, s) in ((w.Lvn, vn, 1.0), (hd, w.Lvn, -m.hyperdiffusion))
        launch!(m, icon_cell_sum_kernel!, (nc, nlev), w.δ, x, m.cell_edges, m.divergence)
        launch!(m, vorticity_kernel!, (nv, nlev), w.ζ, x, m.vertex_edges, m.curl)
        launch!(m, vector_laplacian_kernel!, (ne, nlev), out, w.δ, w.ζ, m.edge_cells, m.edge_vertices,
                m.inverse_dual, m.inverse_primal, s)
    end
    return hd
end
