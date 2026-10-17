"""
    DofDistribution

Ownership of the dofs of one space on one rank: `owned` dofs (this rank
computes and solves them), `local_dofs` (all dofs touched by the local kites,
owned and ghost) and the halo exchange that updates the ghosts.
"""
struct DofDistribution
    owned::Vector{Int}
    local_dofs::Vector{Int}
    halo::HaloExchange
end

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

In this first version every rank keeps the global mesh and global-length
vectors but works only on its local kites and owned dofs.
"""
struct KiteDistribution{C}
    comm::C
    rank::Int
    nranks::Int
    part::Vector{Int}             # part (1-based) per primal cell index
    owned_kites::Vector{Int}      # kite cell indices
    local_kites::Vector{Int}      # owned and ghost kites
    u::DofDistribution
    h::DofDistribution
    z::DofDistribution
end

is_distributed(d::KiteDistribution) = d.nranks > 1

"Serial distribution: one rank owns everything."
function KiteDistribution(primal::Plex, kites::Plex, U, H, Z)
    all_kites = collect(1:num_cells(kites))
    whole(V) = DofDistribution(collect(1:num_dofs(V)), collect(1:num_dofs(V)), HaloExchange())
    return KiteDistribution(nothing, 0, 1, ones(Int, num_cells(primal)), all_kites, all_kites, whole(U), whole(H), whole(Z))
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
        owned = sort!([d for d in locals[me] if owner[d] == me])
        neighbors, send, recv = Int[], Vector{Int}[], Vector{Int}[]
        for q in 1:nranks
            q == me && continue
            from_q = [d for d in locals[me] if owner[d] == q]   # my ghosts owned by q
            to_q = [d for d in locals[q] if owner[d] == me]     # q's ghosts owned by me
            isempty(from_q) && isempty(to_q) && continue
            push!(neighbors, q - 1); push!(recv, from_q); push!(send, to_q)
        end
        return DofDistribution(owned, locals[me], HaloExchange(neighbors, send, recv))
    end
    owned_kites = [k for k in 1:nk if kite_part[k] == rank + 1]
    return KiteDistribution(comm, rank, nranks, collect(Int, part), owned_kites, local_kites[rank + 1],
                            distribution(dofs_u, owner_u), distribution(dofs_h, owner_h),
                            distribution(dofs_z, owner_z))
end

exchange!(x::AbstractVector, d::DofDistribution, comm) = exchange!(x, d.halo, comm)

"Sum of `x` over all ranks (identity in serial)."
global_sum(d::KiteDistribution, x::Real) = is_distributed(d) ? MPI.Allreduce(x, +, d.comm) : x

"""
    gather_owned(d, x, space)

Global vector with the owned entries of every rank (zeros elsewhere summed
away); identical on all ranks.
"""
function gather_owned(d::KiteDistribution, x::AbstractVector, space::Symbol)
    is_distributed(d) || return copy(x)
    owned = getfield(d, space).owned
    y = zeros(length(x))
    y[owned] .= x[owned]
    return MPI.Allreduce(y, +, d.comm)
end

"""
    restrict(S, owned)

Mass solver that only solves the blocks of the `owned` dofs (block-diagonal
solvers only).
"""
function restrict(S::MassSolver{BlockInverse}, owned::AbstractVector{<:Integer})
    mine = falses(size(S.matrix, 1))
    mine[owned] .= true
    keep = [i for (i, block) in enumerate(S.factor.indices) if mine[first(block)]]
    return MassSolver(S.matrix, BlockInverse(S.factor.indices[keep], S.factor.inverses[keep]))
end

restrict(::MassSolver, owned) = throw(ArgumentError("distributed solves need a block-diagonal mass matrix"))
