module AdvectionModule

using ..DMPlexModule
using ..MetricModule

export compute_advection_rhs!

"""
    compute_advection_rhs!(rhs, C, dm, metrics, U_normal)

Berechnet dC/dt rein topologisch über die im DMPlex eingeprägten Vorzeichen-Orientierungen.
Entspricht exakt der High-Performance-Assemblierung in PETSc.
"""
function compute_advection_rhs!(
    rhs::Vector{Float64},
    C::Vector{Float64}, 
    dm::MyDMPlex, 
    metrics::MeshMetrics, 
    U_normal::Vector{Float64}
)
    fill!(rhs, 0.0)

    # Schleife über alle Kanten (Kanten-basiertes Residuum)
    @inbounds for e_point in dm.edge_start:dm.edge_end
        e_idx = e_point - dm.edge_start + 1
        
        length_e = metrics.edge_lengths[e_point]
        u_n = U_normal[e_idx] # Geschwindigkeit relativ zur positiven Kanten-Normale

        # Strikte PETSc Support-Abfrage:
        # cell_L hat die Kante im Graphen als (+), d.h. u_n strömt hier HERAUS.
        # cell_R hat die Kante im Graphen als (-), d.h. u_n strömt hier HINEIN.
        cell_L, cell_R = get_support_cells(dm, e_point)
        if cell_L == 0 || cell_R == 0 continue end # Sicherheitscheck für ungeschlossene Ränder

        cL_idx = cell_L - dm.cell_start + 1
        cR_idx = cell_R - dm.cell_start + 1

        # --- REINE TOPOLOGISCHE UPWIND ENTSCHEIDUNG ---
        if u_n > 0.0
            flux = length_e * u_n * C[cL_idx]
        else
            flux = length_e * u_n * C[cR_idx]
        end

        # Fluss-Bilanzierung auf den Zellen
        rhs[cL_idx] -= flux
        rhs[cR_idx] += flux
    end

    # In-place Normierung über das Zellvolumen
    @inbounds for cell_point in dm.cell_start:dm.cell_end
        c_idx = cell_point - dm.cell_start + 1
        volume = metrics.cell_volumes[cell_point]
        rhs[c_idx] /= volume
    end

    return rhs
end

end # module

