"""
    HaloExchange

Communication plan of one dof space on one rank: for every neighbour rank the
owned dofs it needs (`send`) and the ghost dofs it owns (`recv`), with
preallocated buffers. Dof indices are local. Index vectors and buffers live on
the backend of the dof vectors (`to_backend`); device buffers are passed to
MPI directly if it is CUDA-aware and otherwise staged through `host_send` and
`host_recv`.
"""
struct HaloExchange{I<:AbstractVector{Int}, B<:AbstractVector{Float64}}
    neighbors::Vector{Int}           # 0-based ranks
    send::Vector{I}
    recv::Vector{I}
    send_buffers::Vector{B}
    recv_buffers::Vector{B}
    host_send::Vector{Vector{Float64}}   # staging buffers, empty unless needed
    host_recv::Vector{Vector{Float64}}
end

HaloExchange() = HaloExchange(Int[], Vector{Int}[], Vector{Int}[], Vector{Float64}[], Vector{Float64}[],
                              Vector{Float64}[], Vector{Float64}[])

function HaloExchange(neighbors, send, recv)
    return HaloExchange(neighbors, send, recv, [zeros(length(s)) for s in send], [zeros(length(r)) for r in recv],
                        Vector{Float64}[], Vector{Float64}[])
end

function FEM.to_backend(backend, halo::HaloExchange)
    staged = !(backend isa KernelAbstractions.CPU) && !MPI.has_gpu()
    move(v) = [to_backend(backend, x) for x in v]
    return HaloExchange(halo.neighbors, move(halo.send), move(halo.recv), move(halo.send_buffers),
                        move(halo.recv_buffers), staged ? copy.(halo.send_buffers) : Vector{Float64}[],
                        staged ? copy.(halo.recv_buffers) : Vector{Float64}[])
end

@kernel function pack_kernel!(buffer, @Const(x), @Const(dofs))
    i = @index(Global)
    @inbounds buffer[i] = x[dofs[i]]
end

@kernel function unpack_kernel!(x, @Const(buffer), @Const(dofs))
    i = @index(Global)
    @inbounds x[dofs[i]] = buffer[i]
end

function pack!(buffer, x, dofs)
    isempty(dofs) && return buffer
    backend = get_backend(buffer)
    if backend isa KernelAbstractions.CPU
        @inbounds for (i, dof) in enumerate(dofs)
            buffer[i] = x[dof]
        end
    else
        pack_kernel!(backend)(buffer, x, dofs; ndrange=length(dofs))
        KernelAbstractions.synchronize(backend)
    end
    return buffer
end

function unpack!(x, buffer, dofs)
    isempty(dofs) && return x
    backend = get_backend(buffer)
    if backend isa KernelAbstractions.CPU
        @inbounds for (i, dof) in enumerate(dofs)
            x[dof] = buffer[i]
        end
    else
        unpack_kernel!(backend)(x, buffer, dofs; ndrange=length(dofs))
        KernelAbstractions.synchronize(backend)
    end
    return x
end

"""
    exchange!(x, halo, comm)

Overwrite the ghost entries of `x` with the values of their owners.
"""
function exchange!(x::AbstractVector, halo::HaloExchange, comm)
    isempty(halo.neighbors) && return x
    staged = !isempty(halo.host_send)
    requests = MPI.Request[]
    for (k, q) in enumerate(halo.neighbors)
        buffer = staged ? halo.host_recv[k] : halo.recv_buffers[k]
        push!(requests, MPI.Irecv!(buffer, comm; source=q, tag=0))
    end
    for (k, q) in enumerate(halo.neighbors)
        pack!(halo.send_buffers[k], x, halo.send[k])
        buffer = staged ? copyto!(halo.host_send[k], halo.send_buffers[k]) : halo.send_buffers[k]
        push!(requests, MPI.Isend(buffer, comm; dest=q, tag=0))
    end
    MPI.Waitall(requests)
    for (k, dofs) in enumerate(halo.recv)
        staged && copyto!(halo.recv_buffers[k], halo.host_recv[k])
        unpack!(x, halo.recv_buffers[k], dofs)
    end
    return x
end
