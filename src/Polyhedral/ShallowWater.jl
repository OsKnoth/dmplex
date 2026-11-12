# The rotating shallow-water equations on one layer of polyhedral cells (POLYEDER `TaskShallowWater`)

"""
    PolyShallowWater(cx; gravity=9.80616, omega=7.292e-5, bathymetry=nothing, degree=2, up=normalize,
                     cell_vector=:auto, cell_vector_weight=20, comm=nothing, part=nothing, backend=CPU())

The shallow-water equations on one layer of cells (an `extruded_complex` over the sphere), free
slip at the top and the bottom: the height `h` as cell averages and the normal momentum
`U = (h u)⋅n` at the faces (zero on the boundary faces) in one state vector `[h; U]`.

- mass: `∂ₜh_C = -|C|⁻¹ Σ_F s_CF |F| U_F`, conserved exactly;
- pressure: `-∇(g h²/2)⋅n` and `-g h_F ∇b⋅n` at the faces, gradients and face values from the
  interaction regions (`interaction_operators`, zero normal gradient at the top and the bottom);
- advection and Coriolis at the cells: the cell momentum `m_C` from the normal components
  (`cell_vector = :perot`: `cell_vector_matrices`, POLYEDER's `SumNormal`, second order on hexagons
  and quadrilaterals but first order on triangles; `:linear`: `linear_vector_reconstruction`, the
  faces of the cell weighted by `cell_vector_weight` against those of its neighbours, second order
  on triangles, unstable without the weight and less accurate on hexagons; `:auto`: `:linear` on
  triangular prisms, else `:perot`), its Cartesian components transported with the upwind face
  values of the polynomial reconstruction (`PolyReconstruction`, tangential basis, the face
  velocity `U_F/h_F`), plus `-f r̂ × m_C` with `f = 2 Ω⋅r̂`; the cell tendency projected to the
  faces with the face values of the interaction regions (exact for linear fields).

The operators are sparse matrices of the global complex, restricted on every rank to padded
tables in local numbering. Every term is a KernelAbstractions kernel gathering over these
tables (no atomics); with `backend` (`CUDABackend()` after `using CUDA`) the tables, the state and
the work arrays live on the device. With an MPI communicator `comm` the cells are partitioned
(`equal_area_partition` of the centroids unless `part` is given), every face is owned by the
smallest part of its cells, and the ghosts of a rank are exactly the cells and faces in the
stencils of its owned rows. `tendency!` exchanges the state, the cell momentum and the cell
tendency once each; the local state is `[h; U]` over the local cells and faces, owned ones
first, the ghost entries of `dy` are zero. `initial_state` and `gather_state` convert from and to
the global state, which serially is the local one.
"""
struct PolyShallowWater{O <: NamedTuple, W <: NamedTuple, D, H, C, BE}
    complex::PolyComplex
    geometry::PolyGeometry                     # of the global complex, on the host
    cell_vector::Symbol
    interior::Vector{Bool}                     # global faces with two cells
    cells::D                                   # DofDistribution of the cells
    faces::D
    state::H                                   # halo exchange of [h; U]
    cell_halo::H                               # halo exchange of cell fields (3 × cells)
    comm::C
    rank::Int
    nranks::Int
    part::Vector{Int}                          # part of every global cell
    ops::O                                     # tables on the backend
    work::W
    gravity::Float64
    backend::BE
end

is_distributed(m::PolyShallowWater) = m.nranks > 1
global_sum(m::PolyShallowWater, x::Real) = is_distributed(m) ? MPI.Allreduce(x, +, m.comm) : x

function PolyShallowWater(cx::PolyComplex; gravity::Real=9.80616, omega::Real=7.292e-5, bathymetry=nothing,
                          degree::Integer=2, up=normalize, cell_vector::Symbol=:auto, cell_vector_weight::Real=20.0,
                          geometry=PolyGeometry(cx), comm=nothing, part=nothing, backend=KernelAbstractions.CPU())
    cell_vector in (:auto, :linear, :perot) || throw(ArgumentError("cell_vector must be :auto, :linear or :perot"))
    if cell_vector == :auto
        triangles = all(c -> length(cell_cone(cx, c)[1]) == 5, 1:num_poly_cells(cx))
        cell_vector = triangles ? :linear : :perot
    end
    nc, nf = num_poly_cells(cx), num_poly_faces(cx)
    # the operators of the global complex
    G, T = interaction_operators(cx, geometry)
    Q = face_value_matrices(cx, PolyReconstruction(cx, geometry, degree; basis=:tangential, up))
    R = cell_vector == :linear ? linear_vector_reconstruction(cx, geometry; up, own_weight=cell_vector_weight) :
        cell_vector_matrices(cx, geometry)
    fc = face_cells(cx)
    interior = [fc[1, f] > 0 && fc[2, f] > 0 for f in 1:nf]
    b = bathymetry === nothing ? zeros(nc) : [bathymetry(x) for x in geometry.cell_centroid]
    gradb = G * b
    # partition, owners and the local cells and faces of every part
    nranks, rank = comm === nothing ? (1, 0) : (MPI.Comm_size(comm), MPI.Comm_rank(comm))
    part = part !== nothing ? collect(Int, part) :
           nranks == 1 ? ones(Int, nc) : equal_area_partition(geometry.cell_centroid, nranks)
    face_owner = [minimum(part[c] for c in view(fc, :, f) if c > 0) for f in 1:nf]
    rows(A) = sparse(A')                       # the rows of A as columns
    Tr, Gr, Q1r, Q2r = rows(T), rows(G), rows(Q[1]), rows(Q[2])
    Rr = map(rows, R)
    columns(A, i) = view(A.rowval, nzrange(A, i))
    cell_faces = [cell_cone(cx, c) for c in 1:nc]
    locals = map(1:nranks) do p
        owned_cells = findall(==(p), part)
        flux_faces = sort!(unique!(Int[f for c in owned_cells for f in cell_faces[c][1]]))
        cells, faces = Set(owned_cells), Set(flux_faces)
        for f in flux_faces
            interior[f] || continue
            foreach(A -> union!(cells, columns(A, f)), (Tr, Q1r, Q2r))
            face_owner[f] == p && union!(cells, columns(Gr, f))
        end
        for c in owned_cells, A in Rr
            union!(faces, columns(A, c))
        end
        (sort!(collect(cells)), sort!(collect(faces)), flux_faces)
    end
    cells = DofDistribution(first.(locals), part, rank)
    faces = DofDistribution(getindex.(locals, 2), face_owner, rank)
    nco, nfo, ncl, nfl = cells.n_owned, faces.n_owned, num_local(cells), num_local(faces)
    local_cell, local_face = zeros(Int, nc), zeros(Int, nf)
    local_cell[cells.global_ids] .= 1:ncl
    local_face[faces.global_ids] .= 1:nfl
    # the interior faces with fluxes (those of the owned cells), the owned ones first; the boundary
    # faces at the top and the bottom have no flux and no tendency
    inner = filter(f -> interior[faces.global_ids[f]], 1:nfl)
    ghosts = filter(>(nfo), local_face[filter(f -> interior[f], locals[rank + 1][3])])
    flux_faces = [filter(<=(nfo), inner); sort!(ghosts)]
    n_owned_flux = count(<=(nfo), inner)
    position = zeros(Int, nfl)
    position[flux_faces] .= eachindex(flux_faces)
    global_flux = faces.global_ids[flux_faces]
    owned_cells, owned_flux = cells.global_ids[1:nco], global_flux[1:n_owned_flux]
    # tables
    Rcols, Rw = ell_table(Rr, owned_cells, local_face)
    Tcols, (Tw,) = ell_table((Tr,), global_flux, local_cell)
    Qcols, (Q1w, Q2w) = ell_table((Q1r, Q2r), global_flux, local_cell)
    Gcols, (Gw,) = ell_table((Gr,), owned_flux, local_cell)
    k = maximum(c -> length(cell_faces[c][1]), owned_cells; init=1)
    cfaces, csigns = ones(Int32, k, nco), zeros(k, nco)
    for (i, c) in enumerate(owned_cells), (j, (f, s)) in enumerate(zip(cell_faces[c]...))
        position[local_face[f]] > 0 || continue
        cfaces[j, i], csigns[j, i] = position[local_face[f]], s
    end
    xc = geometry.cell_centroid[owned_cells]
    columns3(v) = reduce(hcat, v; init=zeros(3, 0))
    dev(x) = to_backend(backend, x)
    ops = (R_cols=dev(Rcols), R=dev(permutedims(cat(Rw...; dims=3), (3, 1, 2))), flux_faces=dev(Int32.(flux_faces)),
           n_owned_flux=n_owned_flux,
           area=dev(geometry.area[global_flux]), T_cols=dev(Tcols), T=dev(Tw), Q_cols=dev(Qcols), Q1=dev(Q1w),
           Q2=dev(Q2w), cell_faces=dev(cfaces), cell_signs=dev(csigns), volume=dev(geometry.volume[owned_cells]),
           coriolis=dev([2omega * normalize(x)[3] for x in xc]), up=dev(columns3([up(x) for x in xc])),
           G_cols=dev(Gcols), G=dev(Gw), normal=dev(columns3(geometry.normal[owned_flux])), gradb=dev(gradb[owned_flux]))
    zero_array(dims...) = KernelAbstractions.zeros(backend, Float64, dims...)
    nff = length(flux_faces)
    work = (mc=zero_array(3, ncl), t=zero_array(3, ncl), hF=zero_array(nff), M=zero_array(nff), F=zero_array(3, nff))
    state = merged_halo(cells.halo, faces.halo, ncl)
    return PolyShallowWater(cx, geometry, cell_vector, interior, to_backend(backend, cells), to_backend(backend, faces),
                            to_backend(backend, state), to_backend(backend, cells.halo), comm, Int(rank), Int(nranks),
                            part, ops, work, Float64(gravity), backend)
end

"""
    ell_table(As, rows, to_local) -> (columns, weights)

Padded tables (entries × rows) of the rows `rows` of matrices given by their transposes `As`
(the rows as columns): the union of their column patterns, mapped to local numbers by
`to_local` (`Int32`), and the weights of every matrix. Padding has the column 1 and the weight 0.
"""
function ell_table(As, rows::AbstractVector{<:Integer}, to_local::Vector{Int})
    patterns = [sort!(unique!(reduce(vcat, (A.rowval[nzrange(A, r)] for A in As)))) for r in rows]
    k = max(1, maximum(length, patterns; init=0))
    cols, W = ones(Int32, k, length(rows)), [zeros(k, length(rows)) for _ in As]
    for (j, p) in enumerate(patterns), (i, c) in enumerate(p)
        to_local[c] > 0 || error("column $c of row $(rows[j]) is not local")
        cols[i, j] = to_local[c]
        for (a, A) in enumerate(As)
            W[a][i, j] = A[c, rows[j]]
        end
    end
    return cols, W
end

num_states(m::PolyShallowWater) = num_local(m.cells) + num_local(m.faces)
height_dofs(m::PolyShallowWater) = 1:num_local(m.cells)
velocity_dofs(m::PolyShallowWater) = num_local(m.cells) .+ (1:num_local(m.faces))

"""
Local state `[h; U]` on the model's backend: `height(x)` at the cell centroids and the normal
momentum `height(x) velocity(x)⋅n` at the centroids of the interior faces.
"""
function initial_state(m::PolyShallowWater, velocity, height)
    geo = m.geometry
    h = [height(geo.cell_centroid[c]) for c in m.cells.global_ids]
    U = map(m.faces.global_ids) do f
        x = geo.face_centroid[f]
        m.interior[f] ? height(x) * dot(velocity(x), geo.normal[f]) : 0.0
    end
    return to_backend(m.backend, [h; U])
end

"Global state `[h; U]` with the owned values of all ranks (identical on every rank, on the host)."
function gather_state(m::PolyShallowWater, y)
    c = m.comm
    return [gather(m.cells, Array(view(y, height_dofs(m))), num_poly_cells(m.complex), c);
            gather(m.faces, Array(view(y, velocity_dofs(m))), num_poly_faces(m.complex), c)]
end

"Total mass `Σ_C |C| h_C` over the owned cells of all ranks."
total_mass(m::PolyShallowWater, y) =
    global_sum(m, dot(m.ops.volume, view(view(y, height_dofs(m)), 1:m.cells.n_owned)))

# --- kernels -------------------------------------------------------------------

@kernel function poly_cell_vector_kernel!(mc, @Const(U), @Const(cols), @Const(w))
    c = @index(Global)
    s1, s2, s3 = 0.0, 0.0, 0.0
    @inbounds for k in axes(cols, 1)
        u = U[cols[k, c]]
        s1 += w[1, k, c] * u
        s2 += w[2, k, c] * u
        s3 += w[3, k, c] * u
    end
    @inbounds mc[1, c], mc[2, c], mc[3, c] = s1, s2, s3
end

@kernel function poly_flux_kernel!(hF, M, F, @Const(h), @Const(U), @Const(mc), @Const(faces), @Const(area),
                                   @Const(tcols), @Const(tw), @Const(qcols), @Const(q1), @Const(q2))
    j = @index(Global)
    @inbounds begin
        s = 0.0
        for k in axes(tcols, 1)
            s += tw[k, j] * h[tcols[k, j]]
        end
        mass = area[j] * U[faces[j]]
        a1, a2, a3 = 0.0, 0.0, 0.0
        for k in axes(qcols, 1)
            c, w = qcols[k, j], mass >= 0 ? q1[k, j] : q2[k, j]
            a1 += w * mc[1, c]
            a2 += w * mc[2, c]
            a3 += w * mc[3, c]
        end
        r = mass / s
        hF[j], M[j] = s, mass
        F[1, j], F[2, j], F[3, j] = r * a1, r * a2, r * a3
    end
end

@kernel function poly_cell_tendency_kernel!(dh, t, @Const(M), @Const(F), @Const(mc), @Const(cfaces), @Const(csigns),
                                            @Const(volume), @Const(coriolis), @Const(up))
    c = @index(Global)
    @inbounds begin
        sm, a1, a2, a3 = 0.0, 0.0, 0.0, 0.0
        for k in axes(cfaces, 1)
            j, s = cfaces[k, c], csigns[k, c]
            sm += s * M[j]
            a1 += s * F[1, j]
            a2 += s * F[2, j]
            a3 += s * F[3, j]
        end
        iv, f = 1 / volume[c], coriolis[c]
        r1, r2, r3 = up[1, c], up[2, c], up[3, c]
        m1, m2, m3 = mc[1, c], mc[2, c], mc[3, c]
        dh[c] = -sm * iv
        t[1, c] = -a1 * iv - f * (r2 * m3 - r3 * m2)
        t[2, c] = -a2 * iv - f * (r3 * m1 - r1 * m3)
        t[3, c] = -a3 * iv - f * (r1 * m2 - r2 * m1)
    end
end

@kernel function poly_momentum_kernel!(dU, @Const(t), @Const(h), @Const(hF), @Const(faces), @Const(tcols), @Const(tw),
                                       @Const(gcols), @Const(gw), @Const(normal), @Const(gradb), gravity)
    f = @index(Global)
    @inbounds begin
        a1, a2, a3 = 0.0, 0.0, 0.0
        for k in axes(tcols, 1)
            c, w = tcols[k, f], tw[k, f]
            a1 += w * t[1, c]
            a2 += w * t[2, c]
            a3 += w * t[3, c]
        end
        p = 0.0
        for k in axes(gcols, 1)
            hc = h[gcols[k, f]]
            p += gw[k, f] * hc * hc
        end
        dU[faces[f]] = normal[1, f] * a1 + normal[2, f] * a2 + normal[3, f] * a3 - gravity / 2 * p -
                       gravity * hF[f] * gradb[f]
    end
end

"Run `kernel` with `n` work items on the model's backend (waited for on the CPU, stream order on GPUs)."
function launch!(m::PolyShallowWater, kernel, n, args...)
    n == 0 && return nothing
    kernel(m.backend)(args...; ndrange=n)
    m.backend isa KernelAbstractions.CPU && KernelAbstractions.synchronize(m.backend)
    return nothing
end

"The cell momentum `m_C` (3 × local cells, `m.work.mc`, ghosts exchanged) of the state `y` (its ghosts updated)."
function cell_momentum!(m::PolyShallowWater, y)
    exchange!(y, m.state, m.comm)
    return local_cell_momentum!(m, y)
end

function local_cell_momentum!(m::PolyShallowWater, y)
    o, w = m.ops, m.work
    launch!(m, poly_cell_vector_kernel!, m.cells.n_owned, w.mc, view(y, velocity_dofs(m)), o.R_cols, o.R)
    exchange!(w.mc, m.cell_halo, m.comm)
    return w.mc
end

function tendency!(dy, y, m::PolyShallowWater)
    o, w = m.ops, m.work
    exchange!(y, m.state, m.comm)
    h, U = view(y, height_dofs(m)), view(y, velocity_dofs(m))
    dh, dU = view(dy, height_dofs(m)), view(dy, velocity_dofs(m))
    local_cell_momentum!(m, y)
    launch!(m, poly_flux_kernel!, length(w.hF), w.hF, w.M, w.F, h, U, w.mc, o.flux_faces, o.area, o.T_cols, o.T,
            o.Q_cols, o.Q1, o.Q2)
    launch!(m, poly_cell_tendency_kernel!, m.cells.n_owned, dh, w.t, w.M, w.F, w.mc, o.cell_faces, o.cell_signs,
            o.volume, o.coriolis, o.up)
    exchange!(w.t, m.cell_halo, m.comm)
    fill!(dU, 0.0)
    launch!(m, poly_momentum_kernel!, o.n_owned_flux, dU, w.t, h, w.hF, o.flux_faces, o.T_cols, o.T, o.G_cols, o.G,
            o.normal, o.gradb, m.gravity)
    fill!(view(dh, (m.cells.n_owned + 1):length(dh)), 0.0)
    return dy
end

"Time step of the gravity waves and the flow, `cfl × min_C |C| / Σ_F |F| (|u_n| + √(g h_F))`, over all ranks."
function shallow_water_time_step(m::PolyShallowWater, y; cfl::Real=0.4)
    tendency!(similar(y), y, m)
    U = Array(view(y, velocity_dofs(m)))
    hF, faces, area = Array(m.work.hF), Array(m.ops.flux_faces), Array(m.ops.area)
    cfaces, signs, volume = Array(m.ops.cell_faces), Array(m.ops.cell_signs), Array(m.ops.volume)
    dt = Inf
    for c in axes(cfaces, 2)
        s = 0.0
        for k in axes(cfaces, 1)
            j = cfaces[k, c]
            signs[k, c] == 0 && continue
            s += area[j] * (abs(U[faces[j]]) / hF[j] + sqrt(m.gravity * hF[j]))
        end
        s > 0 && (dt = min(dt, volume[c] / s))
    end
    dt = is_distributed(m) ? MPI.Allreduce(dt, min, m.comm) : dt
    return cfl * dt
end
