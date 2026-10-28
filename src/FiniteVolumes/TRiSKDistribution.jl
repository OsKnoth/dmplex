"""
    TRiSKDistribution(grid, comm; part=equal_area_partition(grid.dm, size))
    TRiSKDistribution(grid)                       # serial

Distribution of a TRiSK C grid over MPI ranks. Cells are partitioned (equal
area by default); every rank holds its owned cells plus two vertex rings of
ghost cells, and all edges and vertices of these cells. Edges and vertices are
owned by the smallest part among their cells, so every owned edge has an owned
cell. With two rings the whole stencil of the owned tendencies is local: the
state `[u; h]` is exchanged once per tendency, the intermediate quantities
(kinetic energy, mass flux, PV) are computed on all local entities and the
tendencies on the owned ones.

Cells and edges are `DofDistribution`s (local numbering with the owned ones
first, halo exchange); vertices carry no state and are only numbered, owned
first. `state` is the halo exchange of the local state `[u; h]` with one
message per neighbour (its edges, then its cells). `comm === nothing` is the
serial case (everything owned, no halo).
"""
struct TRiSKDistribution{C, DC, DE, H}
    comm::C
    rank::Int
    nranks::Int
    part::Vector{Int}            # part (1-based) per global cell
    cells::DC
    edges::DE
    state::H
    vertex_ids::Vector{Int}      # local vertex → global vertex, owned first
    n_owned_vertices::Int
end

TRiSKDistribution(comm, rank, nranks, part, cells, edges, vertex_ids, n_owned_vertices) =
    TRiSKDistribution(comm, rank, nranks, part, cells, edges, merged_halo(edges.halo, cells.halo, num_local(edges)),
                      vertex_ids, n_owned_vertices)

FEM.to_backend(backend, d::TRiSKDistribution) =
    TRiSKDistribution(d.comm, d.rank, d.nranks, d.part, to_backend(backend, d.cells), to_backend(backend, d.edges),
                      to_backend(backend, d.state), d.vertex_ids, d.n_owned_vertices)

"""
Halo exchange of the concatenation `[x; y]` from those of `x` (`a`) and `y`
(`b`, indices shifted by `offset`): one message per neighbour.
"""
function merged_halo(a::HaloExchange, b::HaloExchange, offset::Integer)
    neighbors = sort!(union(a.neighbors, b.neighbors))
    list(h, q, which) = (k = findfirst(==(q), h.neighbors); k === nothing ? Int[] : getfield(h, which)[k])
    send = [[list(a, q, :send); list(b, q, :send) .+ offset] for q in neighbors]
    recv = [[list(a, q, :recv); list(b, q, :recv) .+ offset] for q in neighbors]
    return isempty(neighbors) ? HaloExchange() : HaloExchange(neighbors, send, recv)
end

is_distributed(d::TRiSKDistribution) = d.nranks > 1
global_sum(d::TRiSKDistribution, x::Real) = is_distributed(d) ? MPI.Allreduce(x, +, d.comm) : x

function TRiSKDistribution(g::TRiSKGrid)
    whole(n) = DofDistribution(collect(1:n), n, HaloExchange())
    return TRiSKDistribution(nothing, 0, 1, ones(Int, num_cells(g)), whole(num_cells(g)), whole(num_edges(g)),
                             collect(1:num_vertices(g)), num_vertices(g))
end

function TRiSKDistribution(g::TRiSKGrid, comm; part=nothing)
    comm === nothing && return TRiSKDistribution(g)
    nranks, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    part = part === nothing ? equal_area_partition(g.dm, nranks; centers=g.centers) : collect(Int, part)
    return TRiSKDistribution(g, part, nranks, rank; comm)
end

"Distribution of `rank` (0-based) for a given partition, also without MPI (for tests)."
function TRiSKDistribution(g::TRiSKGrid, part::AbstractVector{<:Integer}, nranks::Integer, rank::Integer;
                           comm=nothing)
    nc, ne, nv = num_cells(g), num_edges(g), num_vertices(g)
    corners(i) = (g.cell_vertices[k, i] for k in axes(g.cell_vertices, 1) if g.cell_kites[k, i] > 0)
    sides(i) = (g.cell_edges[k, i] for k in axes(g.cell_edges, 1) if g.cell_signs[k, i] != 0)
    cells_of_vertex = [Int[] for _ in 1:nv]
    for i in 1:nc, v in corners(i)
        push!(cells_of_vertex[v], i)
    end
    ring(cells) = union(cells, (c for i in cells for v in corners(i) for c in cells_of_vertex[v]))
    local_cells = [sort!(collect(Int, ring(ring(findall(==(p), part))))) for p in 1:nranks]
    local_edges = [sort!(unique!(Int[e for i in cells for e in sides(i)])) for cells in local_cells]

    edge_owner = [min(part[g.edge_cells[1, e]], part[g.edge_cells[2, e]]) for e in 1:ne]
    vertex_owner = [minimum(part[c] for c in cells_of_vertex[v]) for v in 1:nv]
    me = rank + 1
    vertices = unique!(Int[v for i in local_cells[me] for v in corners(i)])
    vertex_ids = [filter(v -> vertex_owner[v] == me, vertices); filter(v -> vertex_owner[v] != me, vertices)]
    return TRiSKDistribution(comm, Int(rank), Int(nranks), collect(Int, part),
                             DofDistribution(local_cells, part, rank), DofDistribution(local_edges, edge_owner, rank),
                             vertex_ids, count(v -> vertex_owner[v] == me, vertices))
end

"""
    local_table(indices, weights, columns, to_local) -> (indices, weights)

Columns `columns` of a padded neighbour table with the indices mapped to local
numbers by `to_local`; neighbours that are not local get index `1` and
weight `0` (they only occur for entities at the outer edge of the halo, whose
values are never used for owned tendencies).
"""
function local_table(indices::AbstractMatrix{Int}, weights, columns, to_local::Vector{Int})
    I = indices[:, columns]
    W = weights === nothing ? nothing : weights[:, columns]
    for k in eachindex(I)
        local_index = to_local[I[k]]
        if local_index == 0
            I[k] = 1
            W === nothing || (W[k] = 0.0)
        else
            I[k] = local_index
        end
    end
    return I, W
end

"Global → local map (0 if not local)."
function local_numbers(global_ids::AbstractVector{Int}, n::Integer)
    to_local = zeros(Int, n)
    to_local[global_ids] .= eachindex(global_ids)
    return to_local
end
