"""
    DofDistribution

Local numbering of the dofs of one space on one rank: local dof `i` is global
dof `global_ids[i]`; the first `n_owned` local dofs are owned (this rank
computes and solves them), the rest are ghosts, updated by the `halo`
exchange (in local indices).
"""
struct DofDistribution{H<:HaloExchange}
    global_ids::Vector{Int}
    n_owned::Int
    halo::H
end

FEM.to_backend(backend, d::DofDistribution) = DofDistribution(d.global_ids, d.n_owned, to_backend(backend, d.halo))

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
struct KiteDistribution{C, DU, DH, DZ}
    comm::C
    rank::Int
    nranks::Int
    part::Vector{Int}             # part (1-based) per primal cell index
    kite_ids::Vector{Int}         # local kite → global kite index, owned first
    n_owned_kites::Int
    u::DU
    h::DH
    z::DZ
end

FEM.to_backend(backend, d::KiteDistribution) =
    KiteDistribution(d.comm, d.rank, d.nranks, d.part, d.kite_ids, d.n_owned_kites,
                     to_backend(backend, d.u), to_backend(backend, d.h), to_backend(backend, d.z))

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

    distribution(dofs, owner) = DofDistribution([local_dofs(dofs, p) for p in 1:nranks], owner, rank)
    mine = local_kites[rank + 1]
    owned_ids = [k for k in mine if kite_part[k] == rank + 1]
    kite_ids = [owned_ids; [k for k in mine if kite_part[k] != rank + 1]]
    return KiteDistribution(comm, rank, nranks, collect(Int, part), kite_ids, length(owned_ids),
                            distribution(dofs_u, owner_u), distribution(dofs_h, owner_h),
                            distribution(dofs_z, owner_z))
end

"""
    DofDistribution(locals, owner, rank)

Distribution of the dofs of one space on `rank` (0-based) from the sorted
local dofs `locals[p]` of every part `p` and the owning part `owner[i]` of
every global dof `i`: owned dofs first, then ghosts, with the halo exchange
between the parts.
"""
function DofDistribution(locals::AbstractVector{<:AbstractVector{<:Integer}}, owner::AbstractVector{<:Integer},
                         rank::Integer)
    me = rank + 1
    owned_ids = [d for d in locals[me] if owner[d] == me]
    ghost_ids = [d for d in locals[me] if owner[d] != me]
    global_ids = [owned_ids; ghost_ids]
    to_local = Dict(g => i for (i, g) in enumerate(global_ids))
    neighbors, send, recv = Int[], Vector{Int}[], Vector{Int}[]
    for q in eachindex(locals)
        q == me && continue
        from_q = [to_local[d] for d in locals[me] if owner[d] == q]  # my ghosts owned by q
        to_q = [to_local[d] for d in locals[q] if owner[d] == me]    # q's ghosts owned by me
        isempty(from_q) && isempty(to_q) && continue
        push!(neighbors, q - 1); push!(recv, from_q); push!(send, to_q)
    end
    return DofDistribution(global_ids, length(owned_ids), HaloExchange(neighbors, send, recv))
end

"""
    dof_owners(V, part) -> owner per global dof

Owning part of every dof of `V` for the cell partition `part` (indexed by
`cell_index`): the smallest part among the cells whose closure holds the dof.
"""
function dof_owners(V::FESpace, part::AbstractVector{<:Integer})
    owner = fill(typemax(Int), num_dofs(V))
    for c in cell_range(V.dm), d in first(cell_dofs(V, c))
        owner[d] = min(owner[d], part[cell_index(V.dm, c)])
    end
    return owner
end

"""
    contiguous_numbering(owner) -> (number, offsets)

Global numbering in which the dofs of every part are contiguous, ordered by
part and, within a part, by their global index (the order of the owned dofs of
a `DofDistribution`): global dof `i` becomes `number[i]`, and part `p` owns
`offsets[p] + 1 : offsets[p + 1]`. This is the row distribution of a
distributed sparse solver.
"""
function contiguous_numbering(owner::AbstractVector{<:Integer})
    order = sortperm(owner; alg=Base.Sort.DEFAULT_STABLE)
    number = invperm(order)
    counts = zeros(Int, maximum(owner; init=0))
    foreach(p -> counts[p] += 1, owner)
    return number, [0; cumsum(counts)]
end

"""
    distributed_rows(M, d, owner, rank) -> (T, first_row)

Rows of the owned dofs of `d` (on `rank`, 0-based) of the symmetric global
matrix `M` in the contiguous numbering of `owner`, upper triangle only,
stored transposed (`T` is `n × n_owned`, so its CSC arrays are the CSR arrays
of the rows); the rows are `first_row + 1 : first_row + n_owned`. This is the
input of a row-distributed sparse solver.
"""
function distributed_rows(M::SparseMatrixCSC, d::DofDistribution, owner::AbstractVector{<:Integer}, rank::Integer)
    number, offsets = contiguous_numbering(owner)
    first_row = offsets[rank + 1]
    ids = d.global_ids[owned(d)]
    number[ids] == first_row .+ (1:d.n_owned) ||
        throw(ArgumentError("owned dofs do not match the contiguous numbering of owner"))
    rows, cols, values = findnz(M[ids, :])
    cols = number[cols]
    upper = cols .>= first_row .+ rows
    return sparse(cols[upper], rows[upper], values[upper], size(M, 2), d.n_owned), first_row
end

"""
    distributed_cudss_solver(M, d, owner, comm; structure="SPD", comm_layer=nothing)

Distributed sparse direct solver for the global matrix `M` with cuDSS in
multi-GPU multi-node mode: every rank holds the rows of its owned dofs of the
distribution `d` (`owner` from `DofDistribution(V, cells)`) in the
contiguous numbering, and `ldiv!(x, S, b)` solves for the owned entries of the
local vectors `x`, `b` (ghosts of `x` are updated by halo exchange). `comm_layer`
is the cuDSS communication layer library (default: `ENV["CUDSS_COMM_LIB"]`,
otherwise `libcudss_commlayer_openmpi.so` next to `libcudss`). Needs
`using CUDA, CUDSS` (package extension) and one GPU per rank.
"""
function distributed_cudss_solver end

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
