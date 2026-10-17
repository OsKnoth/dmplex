"""
    HaloExchange

Communication plan of one dof space on one rank: for every neighbour rank the
owned dofs it needs (`send`) and the ghost dofs it owns (`recv`), with
preallocated buffers. Dof indices are global section indices.
"""
struct HaloExchange
    neighbors::Vector{Int}           # 0-based ranks
    send::Vector{Vector{Int}}
    recv::Vector{Vector{Int}}
    send_buffers::Vector{Vector{Float64}}
    recv_buffers::Vector{Vector{Float64}}
end

HaloExchange() = HaloExchange(Int[], Vector{Int}[], Vector{Int}[], Vector{Float64}[], Vector{Float64}[])

function HaloExchange(neighbors, send, recv)
    return HaloExchange(neighbors, send, recv, [zeros(length(s)) for s in send], [zeros(length(r)) for r in recv])
end

"""
    exchange!(x, halo, comm)

Overwrite the ghost entries of `x` with the values of their owners.
"""
function exchange!(x::AbstractVector, halo::HaloExchange, comm)
    isempty(halo.neighbors) && return x
    requests = MPI.Request[]
    for (k, q) in enumerate(halo.neighbors)
        push!(requests, MPI.Irecv!(halo.recv_buffers[k], comm; source=q, tag=0))
    end
    for (k, q) in enumerate(halo.neighbors)
        buffer = halo.send_buffers[k]
        for (i, dof) in enumerate(halo.send[k])
            buffer[i] = x[dof]
        end
        push!(requests, MPI.Isend(buffer, comm; dest=q, tag=0))
    end
    MPI.Waitall(requests)
    for (k, dofs) in enumerate(halo.recv)
        buffer = halo.recv_buffers[k]
        for (i, dof) in enumerate(dofs)
            x[dof] = buffer[i]
        end
    end
    return x
end
