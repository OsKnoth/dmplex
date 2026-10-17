"""
    compute_advection_rhs!(rhs, C, dm, metrics, U_normal)

First-order upwind finite-volume tendency `dC/dt` of a cell-wise constant
tracer. `U_normal[i]` is the normal velocity of edge `i`, positive from its
left to its right cell.
"""
function compute_advection_rhs!(
    rhs::Vector{Float64},
    C::Vector{Float64},
    dm::Plex,
    metrics::MeshMetrics,
    U_normal::Vector{Float64}
)
    fill!(rhs, 0.0)
    for e in edge_range(dm)
        i = edge_index(dm, e)
        left, right = get_edge_cells(dm, e)
        (left == 0 || right == 0) && continue
        l, r = cell_index(dm, left), cell_index(dm, right)
        u_n = U_normal[i]
        flux = metrics.edge_lengths[i] * u_n * (u_n > 0.0 ? C[l] : C[r])
        rhs[l] -= flux
        rhs[r] += flux
    end
    rhs ./= metrics.cell_areas
    return rhs
end
