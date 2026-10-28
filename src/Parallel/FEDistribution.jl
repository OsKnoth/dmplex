"""
    CellDistribution

Distribution of the cells of a mesh over MPI ranks for classical finite
elements: the cells are partitioned (`part`, equal-area by default) and every
rank holds its owned cells plus one vertex ring of ghost cells. Every cell
containing an owned dof is then local, so integrals over the local cells give
complete right-hand sides for the owned dofs. Local cells are numbered owned
first; `cell_ids` maps them to cell indices, `locals[p]` lists the local cells
of every part (needed to build the halo exchange of each space).
"""
struct CellDistribution{C}
    comm::C
    rank::Int
    nranks::Int
    part::Vector{Int}                 # part (1-based) per cell index
    cell_ids::Vector{Int}             # local cell → cell index, owned first
    n_owned_cells::Int
    locals::Vector{Vector{Int}}       # local cells (cell indices) of every part
end

"Serial distribution: one rank holds all cells."
CellDistribution(dm::Plex) =
    CellDistribution(nothing, 0, 1, ones(Int, num_cells(dm)), collect(1:num_cells(dm)), num_cells(dm),
                     [collect(1:num_cells(dm))])

"""
    CellDistribution(dm, comm; part=equal_area_partition(dm, size))

Cells of `dm` distributed over the ranks of `comm` (`comm === nothing`: serial).
"""
function CellDistribution(dm::Plex, comm; part=nothing)
    comm === nothing && return CellDistribution(dm)
    nranks, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    part = part === nothing ? equal_area_partition(dm, nranks) : collect(Int, part)
    return CellDistribution(dm, part, nranks, rank; comm)
end

"Cell distribution of `rank` (0-based) for a given partition, also without MPI (for tests)."
function CellDistribution(dm::Plex, part::AbstractVector{<:Integer}, nranks::Integer, rank::Integer; comm=nothing)
    first_cell = first(cell_range(dm)) - 1
    vertices = [[vertex_index(dm, v) for v in get_cell_vertices(dm, first_cell + c)] for c in 1:num_cells(dm)]
    cells_of_vertex = [Int[] for _ in 1:num_vertices(dm)]
    for (c, vs) in enumerate(vertices), v in vs
        push!(cells_of_vertex[v], c)
    end
    locals = map(1:nranks) do p
        owned_cells = findall(==(p), part)
        ring = Set{Int}()
        for c in owned_cells, v in vertices[c]
            union!(ring, cells_of_vertex[v])
        end
        [owned_cells; sort!([c for c in ring if part[c] != p])]
    end
    me = rank + 1
    return CellDistribution(comm, Int(rank), Int(nranks), collect(Int, part), locals[me], count(==(me), part),
                            locals)
end

owned_cells(d::CellDistribution) = 1:d.n_owned_cells
num_local_cells(d::CellDistribution) = length(d.cell_ids)
is_distributed(d::CellDistribution) = d.nranks > 1
global_sum(d::CellDistribution, x::Real) = is_distributed(d) ? MPI.Allreduce(x, +, d.comm) : x

"""
    DofDistribution(V, cells::CellDistribution) -> (distribution, owner)

Distribution of the dofs of `V` on the local cells (owned and ghost) of
`cells`: the local dofs are the dofs of the local cells, each dof is owned by
the smallest part among its cells (`dof_owners`).
"""
function DofDistribution(V::FESpace, cells::CellDistribution)
    owner = dof_owners(V, cells.part)
    first_cell = first(cell_range(V.dm)) - 1
    locals = map(cells.locals) do local_cells
        sort!(unique!(reduce(vcat, (first(cell_dofs(V, first_cell + c)) for c in local_cells); init=Int[])))
    end
    return DofDistribution(locals, owner, cells.rank), owner
end

"""
    GatheredMassSolver

Exact distributed solve with a factorization of the global matrix on every
rank: the owned right-hand sides are gathered (`Allgatherv`, in the contiguous
numbering), every rank solves the global system and keeps its local entries,
owned and ghosts, so no halo exchange is needed. Not scalable, but exact; the
CPU reference for the distributed GPU solver (`distributed_cudss_solver`).
Vectors on a device are staged through host buffers.
"""
struct GatheredMassSolver{S, D, C}
    solver::S
    distribution::D
    comm::C
    number::Vector{Int}       # contiguous number of every global dof
    counts::Vector{Cint}      # owned dofs per rank
    send::Vector{Float64}
    gathered::Vector{Float64}
    rhs::Vector{Float64}
    solution::Vector{Float64}
    x::Vector{Float64}        # local entries of the solution
end

function GatheredMassSolver(S::MassSolver, d::DofDistribution, owner::AbstractVector{<:Integer}, comm)
    number, offsets = contiguous_numbering(owner)
    n = size(S.matrix, 1)
    return GatheredMassSolver(S, d, comm, number, Cint.(diff(offsets)), zeros(d.n_owned), zeros(n), zeros(n), zeros(n),
                              zeros(num_local(d)))
end

function LinearAlgebra.ldiv!(x::AbstractVector, S::GatheredMassSolver, b::AbstractVector)
    d = S.distribution
    copyto!(S.send, 1, b, 1, d.n_owned)
    MPI.Allgatherv!(S.send, MPI.VBuffer(S.gathered, S.counts), S.comm)
    for i in eachindex(S.rhs)
        S.rhs[i] = S.gathered[S.number[i]]
    end
    ldiv!(S.solution, S.solver, S.rhs)
    for (j, g) in enumerate(d.global_ids)
        S.x[j] = S.solution[g]
    end
    return copyto!(x, S.x)
end

"A local solver for the owned dofs followed by the halo exchange of the ghosts."
struct HaloSolver{S, D, C}
    solver::S
    distribution::D
    comm::C
end

function LinearAlgebra.ldiv!(x::AbstractVector, S::HaloSolver, b::AbstractVector)
    ldiv!(x, S.solver, b)
    return exchange!(x, S.distribution, S.comm)
end

"""
    distributed_mass_solver(V, d, owner, comm; solver=:gathered, backend=CPU())

Mass solver of `V` for local vectors of the distribution `d` on `backend`:
`ldiv!(x, S, b)` reads the owned entries of `b` and fills all local entries of
`x`. Block-diagonal mass matrices whose blocks are each owned by one rank (DG,
lumped) are solved locally on the owned blocks (`BlockSolver`) followed by a
halo exchange; coupled ones (RT, CG) with `solver=:gathered`
(`GatheredMassSolver`, host factorization) or `solver=:cudss`
(`distributed_cudss_solver`, GPUs). Serial (`comm === nothing`): the cached
`mass_solver(V)` on the CPU, otherwise its `device_solver`.
"""
function distributed_mass_solver(V::FESpace, d::DofDistribution, owner::AbstractVector{<:Integer}, comm;
                                 solver::Symbol=:gathered, backend=KernelAbstractions.CPU())
    solver in (:gathered, :cudss) || throw(ArgumentError("solver must be :gathered or :cudss"))
    S = mass_solver(V)
    comm === nothing && return device_solver(S, backend; solver=solver == :cudss ? :cudss : :host)
    dd = to_backend(backend, d)
    # blocks owned by one rank each (DG, lumped) are solved locally; a coupled matrix may also be stored
    # as a few large blocks, which span ranks
    S.factor isa BlockInverse && all(b -> allequal(view(owner, b)), S.factor.indices) &&
        return HaloSolver(block_solver(local_solver(S, d), backend), dd, comm)
    solver == :gathered && return GatheredMassSolver(S, d, owner, comm)
    return distributed_cudss_solver(S.matrix, dd, owner, comm)
end
