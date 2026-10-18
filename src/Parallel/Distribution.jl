"""
    DofDistribution

Local numbering of the dofs of one space on one rank: local dof `i` is global
dof `global_ids[i]`; the first `n_owned` local dofs are owned (this rank
computes and solves them), the rest are ghosts, updated by the `halo`
exchange (in local indices).
"""
struct DofDistribution
    global_ids::Vector{Int}
    n_owned::Int
    halo::HaloExchange
end

owned(d::DofDistribution) = 1:d.n_owned
num_local(d::DofDistribution) = length(d.global_ids)

"""
    KiteDistribution

Distribution of a kite model over MPI ranks. Primal cells are partitioned
(equal-area regions), every kite belongs to the rank of its primal cell, and
each rank additionally holds one vertex ring of ghost primal cells. Dof
ownership: height dofs follow their primal cell, velocity dofs their dual-cell
mass block (the smallest part around the dual vertex), vorticity dofs the
smallest part among the kites touching them. With the ghost ring every owned
dof has all its kites locally, so right-hand sides computed over the local
kites are complete for the owned dofs, and all mass blocks of owned dofs are
local.

Kites and dofs are numbered locally, owned ones first; `kite_ids` and the
`global_ids` of the spaces map to the global numbering, which is only needed
during setup and for gathering. (The global mesh is still built on every rank.)
"""
struct KiteDistribution{C}
    comm::C
    rank::Int
    nranks::Int
    part::Vector{Int}             # part (1-based) per primal cell index
    kite_ids::Vector{Int}         # local kite → global kite index, owned first
    n_owned_kites::Int
    u::DofDistribution
    h::DofDistribution
    z::DofDistribution
end

owned_kites(d::KiteDistribution) = 1:d.n_owned_kites
num_local_kites(d::KiteDistribution) = length(d.kite_ids)

is_distributed(d::KiteDistribution) = d.nranks > 1

"Serial distribution: one rank owns everything."
function KiteDistribution(primal::Plex, kites::Plex, U, H, Z)
    whole(V) = DofDistribution(collect(1:num_dofs(V)), num_dofs(V), HaloExchange())
    return KiteDistribution(nothing, 0, 1, ones(Int, num_cells(primal)), collect(1:num_cells(kites)), num_cells(kites),
                            whole(U), whole(H), whole(Z))
end

"""
    KiteDistribution(primal, kites, U, H, Z, comm; part=equal_area_partition(primal, size))

Distribution of the kite spaces `U`, `H`, `Z` over the ranks of `comm`.
"""
function KiteDistribution(primal::Plex, kites::Plex, U, H, Z, comm;
                          part::AbstractVector{<:Integer}=equal_area_partition(primal, MPI.Comm_size(comm)))
    nranks, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    nk = num_cells(kites)
    parent = [get_label_value(kites, "kite_parent_cell", c) for c in cell_range(kites)]
    dual = [get_label_value(kites, "kite_dual_cell", c) for c in cell_range(kites)]
    kite_part = [part[parent[k]] for k in 1:nk]
    dual_owner = fill(typemax(Int), num_vertices(primal))
    for k in 1:nk
        dual_owner[dual[k]] = min(dual_owner[dual[k]], kite_part[k])
    end

    kite_dofs(V) = [first(cell_dofs(V, c)) for c in cell_range(kites)]
    dofs_u, dofs_h, dofs_z = kite_dofs(U), kite_dofs(H), kite_dofs(Z)
    function min_owner(V, dofs)
        owner = fill(typemax(Int), num_dofs(V))
        for k in 1:nk, d in dofs[k]
            owner[d] = min(owner[d], kite_part[k])
        end
        return owner
    end
    owner_u = fill(typemax(Int), num_dofs(U))
    for k in 1:nk, d in dofs_u[k]
        owner_u[d] = dual_owner[dual[k]]
    end
    owner_h, owner_z = min_owner(H, dofs_h), min_owner(Z, dofs_z)

    # local kites of every part: owned primal cells plus one vertex ring of ghosts
    kites_of_cell = [Int[] for _ in 1:num_cells(primal)]
    for k in 1:nk
        push!(kites_of_cell[parent[k]], k)
    end
    cells_of_vertex = [Int[] for _ in 1:num_vertices(primal)]
    for c in cell_range(primal), v in get_cell_vertices(primal, c)
        push!(cells_of_vertex[vertex_index(primal, v)], cell_index(primal, c))
    end
    function local_kites_of(p)
        cells = Set{Int}()
        for c in cell_range(primal)
            part[cell_index(primal, c)] == p || continue
            for v in get_cell_vertices(primal, c)
                union!(cells, cells_of_vertex[vertex_index(primal, v)])
            end
        end
        return sort!(reduce(vcat, (kites_of_cell[c] for c in cells); init=Int[]))
    end
    local_kites = [local_kites_of(p) for p in 1:nranks]
    local_dofs(dofs, p) = sort!(unique!(reduce(vcat, (dofs[k] for k in local_kites[p]); init=Int[])))

    function distribution(dofs, owner)
        locals = [local_dofs(dofs, p) for p in 1:nranks]
        me = rank + 1
        owned_ids = [d for d in locals[me] if owner[d] == me]
        ghost_ids = [d for d in locals[me] if owner[d] != me]
        global_ids = [owned_ids; ghost_ids]
        to_local = Dict(g => i for (i, g) in enumerate(global_ids))
        neighbors, send, recv = Int[], Vector{Int}[], Vector{Int}[]
        for q in 1:nranks
            q == me && continue
            from_q = [to_local[d] for d in locals[me] if owner[d] == q]  # my ghosts owned by q
            to_q = [to_local[d] for d in locals[q] if owner[d] == me]    # q's ghosts owned by me
            isempty(from_q) && isempty(to_q) && continue
            push!(neighbors, q - 1); push!(recv, from_q); push!(send, to_q)
        end
        return DofDistribution(global_ids, length(owned_ids), HaloExchange(neighbors, send, recv))
    end
    mine = local_kites[rank + 1]
    owned_ids = [k for k in mine if kite_part[k] == rank + 1]
    kite_ids = [owned_ids; [k for k in mine if kite_part[k] != rank + 1]]
    return KiteDistribution(comm, rank, nranks, collect(Int, part), kite_ids, length(owned_ids),
                            distribution(dofs_u, owner_u), distribution(dofs_h, owner_h),
                            distribution(dofs_z, owner_z))
end

exchange!(x::AbstractVector, d::DofDistribution, comm) = exchange!(x, d.halo, comm)

"Sum of `x` over all ranks (identity in serial)."
global_sum(d::KiteDistribution, x::Real) = is_distributed(d) ? MPI.Allreduce(x, +, d.comm) : x

"""
    gather(d, x, nglobal, comm)

Global vector of length `nglobal` with the owned entries of every rank
(local vector `x`); identical on all ranks.
"""
function gather(d::DofDistribution, x::AbstractVector, nglobal::Integer, comm)
    y = zeros(nglobal)
    for i in owned(d)
        y[d.global_ids[i]] = x[i]
    end
    return comm === nothing ? y : MPI.Allreduce(y, +, comm)
end

"""
    local_solver(S, d; ghosts=false)

Block-diagonal mass solver `S` (global numbering) restricted to the blocks of
the owned dofs of `d` (with `ghosts=true` to all blocks inside the local dofs),
in local numbering.
"""
function local_solver(S::MassSolver{BlockInverse}, d::DofDistribution; ghosts::Bool=false)
    to_local = Dict(g => i for (i, g) in enumerate(d.global_ids))
    keep = ghosts ? (i -> haskey(to_local, i)) : (i -> get(to_local, i, typemax(Int)) <= d.n_owned)
    indices, inverses = Vector{Int}[], Matrix{Float64}[]
    for (block, inverse) in zip(S.factor.indices, S.factor.inverses)
        all(keep, block) || continue
        push!(indices, [to_local[i] for i in block])
        push!(inverses, inverse)
    end
    n = num_local(d)
    return MassSolver(spzeros(n, n), BlockInverse(indices, inverses))
end

local_solver(::MassSolver, d; ghosts=false) =
    throw(ArgumentError("distributed solves need a block-diagonal mass matrix"))
