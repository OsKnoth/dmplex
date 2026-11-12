# Transport of cell averages with upwind fluxes of the polynomial reconstruction

"""
    stream_function_flux(cx, ψ; G=identity, height=norm, project=normalize) -> Vector

Volume fluxes through the faces (along their normals) of the velocity `u = g(h) ∇ψ × ∇h` of a
stream function `ψ` of the horizontal position `project(x)`, `h = height(x)` the height
(the radius on a sphere, `x[3]` on a plane with `project = x -> SVector(x[1], x[2], 0)`) and `G` an
antiderivative of the weight `g`: the circulations of the vector potential `g(h) ψ ∇h` around the
faces, `Σ ½ (ψ(a) + ψ(b)) (G(h(b)) - G(h(a)))` over the oriented edges `a → b` of each face. Zero
on the faces of constant height, and the outward fluxes of every cell sum to zero exactly. On the
sphere `ψ = Ω⋅r̂` with `G(r) = r³/3` is the solid-body rotation `Ω × x`.
"""
function stream_function_flux(cx::PolyComplex, ψ; G=identity, height=norm, project=normalize)
    return map(1:num_poly_faces(cx)) do f
        xs = face_points(cx, f)
        sum(eachindex(xs)) do i
            a, b = xs[i], xs[mod1(i + 1, length(xs))]
            (ψ(project(a)) + ψ(project(b))) / 2 * (G(height(b)) - G(height(a)))
        end
    end
end

"""
    PolyAdvection(cx, flux; degree=2, basis=:full, rings=nothing, up=normalize)
    PolyAdvection(cx, velocity::Function; ...)

Transport `∂ₜq + ∇⋅(q u) = 0` of the cell averages `q` with upwind fluxes of the polynomial
reconstruction (`PolyReconstruction`); `tendency!(dq, q, m)` for the time steppers of `dmplex`.
With the volume fluxes `flux` through the faces the flux is `U_F q̄_F` with the upwind face average
`q̄_F` (`face_value_matrices`), second order unless `u⋅n` is constant on the face; with the
`velocity` field it is `Σ_g (u⋅n)_g q_g` at the points of `face_quadrature`, each upwinded
(`point_value_matrices`), of the order of the reconstruction.
"""
struct PolyAdvection
    complex::PolyComplex
    geometry::PolyGeometry
    reconstruction::PolyReconstruction
    Q::NTuple{2, SparseMatrixCSC{Float64, Int}}
    cells::Matrix{Int}             # the two cells of each row of Q (a face or a point of it)
    flux::Vector{Float64}          # the volume flux of each row
    values::NTuple{2, Vector{Float64}}
end

function PolyAdvection(cx::PolyComplex, flux; degree::Integer=2, basis::Symbol=:full,
                       rings=nothing, up=normalize, geometry=PolyGeometry(cx))
    rec = PolyReconstruction(cx, geometry, degree; basis, rings, up)
    fc = face_cells(cx)
    if flux isa Function
        quadrature = [face_quadrature(cx, f) for f in 1:num_poly_faces(cx)]
        Q = point_value_matrices(cx, rec, first.(quadrature))
        cells = reduce(hcat, (repeat(fc[:, f], 1, length(quadrature[f][1])) for f in eachindex(quadrature)))
        U = reduce(vcat, ([dot(flux(x), w) for (x, w) in zip(q...)] for q in quadrature))
    else
        Q, cells, U = face_value_matrices(cx, rec), fc, Vector{Float64}(flux)
    end
    return PolyAdvection(cx, geometry, rec, Q, cells, U, (zeros(length(U)), zeros(length(U))))
end

function tendency!(dq, q, m::PolyAdvection)
    q1, q2 = m.values
    mul!(q1, m.Q[1], q)
    mul!(q2, m.Q[2], q)
    fill!(dq, 0.0)
    @inbounds for f in eachindex(m.flux)
        U = m.flux[f]
        c1, c2 = m.cells[1, f], m.cells[2, f]
        (c1 == 0 || c2 == 0) && continue
        F = U * (U >= 0 ? q1[f] : q2[f])
        dq[c1] -= F
        dq[c2] += F
    end
    dq ./= m.geometry.volume
    return dq
end

"Largest stable time step of the first-order upwind scheme, `min_C |C| / Σ_F max(0, outflow)`."
function upwind_time_step(m::PolyAdvection)
    out = zeros(num_poly_cells(m.complex))
    for f in eachindex(m.flux)
        U = m.flux[f]
        c = U >= 0 ? m.cells[1, f] : m.cells[2, f]
        c > 0 && (out[c] += abs(U))
    end
    return minimum(m.geometry.volume ./ max.(out, eps()))
end
