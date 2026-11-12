# The rotating shallow-water equations on one layer of polyhedral cells (POLYEDER `TaskShallowWater`)

"""
    PolyShallowWater(cx; gravity=9.80616, omega=7.292e-5, bathymetry=nothing, degree=2, up=normalize,
                     cell_vector=:auto, projection=:interaction, cell_vector_weight=20)

The shallow-water equations on one layer of cells (an `extruded_complex` over the sphere), free
slip at the top and the bottom: the height `h` as cell averages and the normal momentum
`U = (h u)⋅n` at the faces (zero on the boundary faces) in one state vector `[h; U]`.

- mass: `∂ₜh_C = -|C|⁻¹ Σ_F s_CF |F| U_F`, conserved exactly;
- pressure: `-∇(g h²/2)⋅n` and `-g h_F ∇b⋅n` at the faces, gradients and face values from the
  interaction regions (`interaction_operators`, zero normal gradient at the top and the bottom);
- advection and Coriolis at the cells: the cell momentum `m_C` from the normal components
  (`cell_vector = :perot`: `cell_vectors`, POLYEDER's `SumNormal`, second order on hexagons and
  quadrilaterals but first order on triangles; `:linear`: `linear_vector_reconstruction`, the faces of
  the cell weighted by `cell_vector_weight` against those of its neighbours, second order on
  triangles, unstable without the weight and less accurate on hexagons; `:auto`: `:linear` on
  triangular prisms, else `:perot`), its Cartesian
  components transported with the upwind faces values of the polynomial reconstruction
  (`PolyReconstruction`, tangential basis, the face velocity `U_F/h_F`), plus `-f r̂ × m_C` with
  `f = 2 Ω⋅r̂`; the cell tendency projected to the faces with the face values of the interaction
  regions (`projection = :interaction`, exact for linear fields) or interpolated between the two
  centroids (`:centroids`, `face_normal_components`).

`tendency!(dy, y, m)` for the time steppers of `dmplex`.
"""
struct PolyShallowWater
    complex::PolyComplex
    geometry::PolyGeometry
    reconstruction::PolyReconstruction
    G::SparseMatrixCSC{Float64, Int}           # normal gradient at the faces
    T::SparseMatrixCSC{Float64, Int}           # values at the faces
    Q::NTuple{2, SparseMatrixCSC{Float64, Int}}
    cells::Matrix{Int}
    interior::Vector{Int}                      # the faces with two cells
    gravity::Float64
    coriolis::Vector{Float64}                  # f at the cells
    up::Vector{SVector{3, Float64}}            # r̂ at the cells
    gradb::Vector{Float64}                     # ∇b⋅n at the faces
    R::Union{Nothing, NTuple{3, SparseMatrixCSC{Float64, Int}}}   # cell vectors (nothing: Perot)
    projection::Symbol
end

function PolyShallowWater(cx::PolyComplex; gravity::Real=9.80616, omega::Real=7.292e-5, bathymetry=nothing,
                          degree::Integer=2, up=normalize, cell_vector::Symbol=:auto,
                          projection::Symbol=:interaction, cell_vector_weight::Real=20.0, geometry=PolyGeometry(cx))
    cell_vector in (:auto, :linear, :perot) || throw(ArgumentError("cell_vector must be :auto, :linear or :perot"))
    if cell_vector == :auto
        triangles = all(c -> length(cell_cone(cx, c)[1]) == 5, 1:num_poly_cells(cx))
        cell_vector = triangles ? :linear : :perot
    end
    projection in (:interaction, :centroids) || throw(ArgumentError("projection must be :interaction or :centroids"))
    G, T = interaction_operators(cx, geometry)
    rec = PolyReconstruction(cx, geometry, degree; basis=:tangential, up)
    fc = face_cells(cx)
    r̂ = [up(x) for x in geometry.cell_centroid]
    b = bathymetry === nothing ? zeros(num_poly_cells(cx)) : [bathymetry(x) for x in geometry.cell_centroid]
    return PolyShallowWater(cx, geometry, rec, G, T, face_value_matrices(cx, rec), fc,
                            findall(f -> fc[1, f] > 0 && fc[2, f] > 0, 1:num_poly_faces(cx)), Float64(gravity),
                            [2omega * n[3] for n in r̂], r̂, G * b,
                            cell_vector == :linear ? linear_vector_reconstruction(cx, geometry; up, own_weight=cell_vector_weight) : nothing, projection)
end

num_states(m::PolyShallowWater) = num_poly_cells(m.complex) + num_poly_faces(m.complex)

function tendency!(dy, y, m::PolyShallowWater)
    cx, geo = m.complex, m.geometry
    nc = num_poly_cells(cx)
    h, U = view(y, 1:nc), view(y, (nc + 1):length(y))
    dh, dU = view(dy, 1:nc), view(dy, (nc + 1):length(dy))
    g = m.gravity
    hF = m.T * h
    # cell momentum, its transport and the Coriolis force
    mc = m.R === nothing ? cell_vectors(cx, geo, U) : SVector.(m.R[1] * U, m.R[2] * U, m.R[3] * U)
    comp = [getindex.(mc, k) for k in 1:3]
    tend = [zeros(nc) for _ in 1:3]
    fill!(dh, 0.0)
    for k in 1:3
        q1, q2 = m.Q[1] * comp[k], m.Q[2] * comp[k]
        for f in m.interior
            c1, c2 = m.cells[1, f], m.cells[2, f]
            M = geo.area[f] * U[f]                       # mass flux
            F = M / hF[f] * (M >= 0 ? q1[f] : q2[f])
            tend[k][c1] -= F
            tend[k][c2] += F
            if k == 1
                dh[c1] -= M
                dh[c2] += M
            end
        end
    end
    dh ./= geo.volume
    cell_tendency = map(1:nc) do c
        SVector(tend[1][c], tend[2][c], tend[3][c]) / geo.volume[c] - m.coriolis[c] * cross(m.up[c], mc[c])
    end
    if m.projection == :interaction
        tf = [m.T * getindex.(cell_tendency, k) for k in 1:3]
        dU .= [dot(geo.normal[f], SVector(tf[1][f], tf[2][f], tf[3][f])) for f in eachindex(dU)]
    else
        dU .= face_normal_components(cx, geo, cell_tendency)
    end
    # pressure gradient and bathymetry
    dU .-= m.G * (g / 2 .* h .^ 2) .+ g .* hF .* m.gradb
    mask = trues(length(dU))
    mask[m.interior] .= false
    dU[mask] .= 0.0
    return dy
end

"Time step of the gravity waves and the flow, `cfl × min_C |C| / Σ_F |F| (|u_n| + √(g h))`."
function shallow_water_time_step(m::PolyShallowWater, y; cfl::Real=0.4)
    cx, geo = m.complex, m.geometry
    nc = num_poly_cells(cx)
    h, U = y[1:nc], y[(nc + 1):end]
    hF = m.T * h
    s = zeros(nc)
    for f in m.interior
        speed = geo.area[f] * (abs(U[f]) / hF[f] + sqrt(m.gravity * max(hF[f], 0.0)))
        s[m.cells[1, f]] += speed
        s[m.cells[2, f]] += speed
    end
    return cfl * minimum(geo.volume ./ s)
end
