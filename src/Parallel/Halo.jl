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

@kernel function pack_layers_kernel!(buffer, @Const(x), @Const(dofs))
    k, i = @index(Global, NTuple)
    @inbounds buffer[k + (i - 1) * size(x, 1)] = x[k, dofs[i]]
end

@kernel function unpack_layers_kernel!(x, @Const(buffer), @Const(dofs))
    k, i = @index(Global, NTuple)
    @inbounds x[k, dofs[i]] = buffer[k + (i - 1) * size(x, 1)]
end

function pack_layers!(buffer, x, dofs)
    isempty(dofs) && return buffer
    backend = get_backend(buffer)
    if backend isa KernelAbstractions.CPU
        L = size(x, 1)
        @inbounds for (i, dof) in enumerate(dofs), k in axes(x, 1)
            buffer[k + (i - 1) * L] = x[k, dof]
        end
    else
        pack_layers_kernel!(backend)(buffer, x, dofs; ndrange=(size(x, 1), length(dofs)))
        KernelAbstractions.synchronize(backend)
    end
    return buffer
end

function unpack_layers!(x, buffer, dofs)
    isempty(dofs) && return x
    backend = get_backend(buffer)
    if backend isa KernelAbstractions.CPU
        L = size(x, 1)
        @inbounds for (i, dof) in enumerate(dofs), k in axes(x, 1)
            x[k, dof] = buffer[k + (i - 1) * L]
        end
    else
        unpack_layers_kernel!(backend)(x, buffer, dofs; ndrange=(size(x, 1), length(dofs)))
        KernelAbstractions.synchronize(backend)
    end
    return x
end

# buffers of the layered exchange, per halo and number of layers
const LAYER_BUFFERS = IdDict{Any, Any}()

function layer_buffers(halo::HaloExchange, L)
    return get!(LAYER_BUFFERS, (halo, L)) do
        staged = !isempty(halo.host_send)
        dev(b, n) = similar(b, n * L)
        (send=[dev(b, length(b)) for b in halo.send_buffers], recv=[dev(b, length(b)) for b in halo.recv_buffers],
         host_send=staged ? [zeros(length(b) * L) for b in halo.send_buffers] : Vector{Float64}[],
         host_recv=staged ? [zeros(length(b) * L) for b in halo.recv_buffers] : Vector{Float64}[])
    end
end

"""
    exchange!(X::AbstractMatrix, halo, comm)

Overwrite the ghost columns of `X` (a 3D field layers first, `X[layer, dof]`) in
every layer with the values of their owners, one message per neighbour for all layers.
"""
function exchange!(X::AbstractMatrix, halo::HaloExchange, comm)
    isempty(halo.neighbors) && return X
    B = layer_buffers(halo, size(X, 1))
    staged = !isempty(B.host_send)
    requests = MPI.Request[]
    for (k, q) in enumerate(halo.neighbors)
        push!(requests, MPI.Irecv!(staged ? B.host_recv[k] : B.recv[k], comm; source=q, tag=1))
    end
    for (k, q) in enumerate(halo.neighbors)
        pack_layers!(B.send[k], X, halo.send[k])
        push!(requests, MPI.Isend(staged ? copyto!(B.host_send[k], B.send[k]) : B.send[k], comm; dest=q, tag=1))
    end
    MPI.Waitall(requests)
    for (k, dofs) in enumerate(halo.recv)
        staged && copyto!(B.recv[k], B.host_recv[k])
        unpack_layers!(X, B.recv[k], dofs)
    end
    return X
end
