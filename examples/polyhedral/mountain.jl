# Compressible Euler on a Cartesian cut-cell complex: flow over an Agnesi hill (Gallus and Klemp 2000)
# cell densities ρ, ρθ, normal momentum (ρu)⋅n at the faces; perturbations of a hydrostatic state
#   julia -t 8 --project=. examples/polyhedral/mountain.jl [hours] [prefix]
# writes θ, w, u on the uncut Cartesian grid every hour to prefix_HHh.bin (plot_mountain.py)
using dmplex, LinearAlgebra, SparseArrays, StaticArrays, Printf
PH = dmplex.Polyhedral

const g, Rd, cp = 9.81, 287.0, 1004.0
const cv, γ, p00 = cp - Rd, cp / (cp - Rd), 1.0e5

Base.@kwdef struct Setup
    h0::Float64 = 400.0; a::Float64 = 1000.0; U0::Float64 = 10.0; N::Float64 = 0.01; θs::Float64 = 300.0
    L::Float64 = 25e3; H::Float64 = 16e3; dx::Float64 = 250.0; dz::Float64 = 125.0
    zD::Float64 = 10e3; xD::Float64 = 15e3; α::Float64 = 0.05      # sponge above zD, beyond |x| > xD
    fraction::Float64 = 0.5
end

zs(s, x) = s.h0 / (1 + (x / s.a)^2)
θ0(s, z) = s.θs * exp(s.N^2 * z / g)
exner0(s, z) = 1 + g^2 / (cp * s.θs * s.N^2) * (exp(-s.N^2 * z / g) - 1)
ρ0(s, z) = p00 * exner0(s, z)^(cv / Rd) / (Rd * θ0(s, z))
pressure(Θ) = p00 * (Rd * Θ / p00)^γ

function cartesian(s)
    nx, nz = round(Int, 2s.L / s.dx), round(Int, s.H / s.dz)
    id(i, j) = (j - 1) * (nx + 1) + i
    pts = [SVector(-s.L + (i - 1) * s.dx, (j - 1) * s.dx, 0.0) for j in 1:2 for i in 1:(nx + 1)]
    polys = [[id(i, 1), id(i + 1, 1), id(i + 1, 2), id(i, 2)] for i in 1:nx]
    return extruded_complex(ExtrudedPlex(plex_from_polygons(polys, pts), nz), collect(range(0.0, s.H; length=nz + 1))), nx, nz
end

struct Model{W <: NamedTuple}
    s::Setup
    cx::PolyComplex
    geo::PolyGeometry
    fc::Matrix{Int}
    cells::Vector{Vector{Tuple{Int, Int}}}   # faces and signs of every cell
    lateral::Vector{Bool}                    # faces at x = ±L, fixed flux
    R::NTuple{3, SparseMatrixCSC{Float64, Int}}
    Q::NTuple{2, SparseMatrixCSC{Float64, Int}}
    dist::Vector{Float64}                    # n⋅(x₂ - x₁) of the interior faces
    w1::Vector{Float64}
    ρref::Vector{Float64}
    pref::Vector{Float64}
    Θref::Vector{Float64}
    Mref::Vector{Float64}
    θin::Vector{Float64}                     # inflow θ at the lateral faces
    uin::Vector{Float64}
    zin::Vector{Float64}
    τc::Vector{Float64}
    τf::Vector{Float64}
    work::W
end

sponge(s, x) = s.α * (x[3] > s.zD ? sin(π / 2 * (x[3] - s.zD) / (s.H - s.zD))^2 : 0.0) +
               s.α * (abs(x[1]) > s.xD ? sin(π / 2 * (abs(x[1]) - s.xD) / (s.L - s.xD))^2 : 0.0)

function Model(s::Setup)
    cx0, nx, nz = cartesian(s)
    cut, parent, terrain = cut_complex(cx0, x -> x[3] - zs(s, x[1]))
    g0 = PolyGeometry(cx0)
    cx, group = merge_small_cells(cut, g0.volume[parent]; fraction=s.fraction)
    geo = PolyGeometry(cx)
    nc, nf = num_poly_cells(cx), num_poly_faces(cx)
    fc = face_cells(cx)
    cells = [collect(zip(PH.cell_cone(cx, c)...)) for c in 1:nc]
    lateral = [(fc[1, f] == 0 || fc[2, f] == 0) && abs(geo.normal[f][1]) > 0.99 && abs(abs(geo.face_centroid[f][1]) - s.L) < 1 for f in 1:nf]
    R = map(A -> sparse(A'), PH.cell_vector_matrices(cx, geo))
    Q = map(A -> sparse(A'), face_value_matrices(cx, PolyReconstruction(cx, geo, 2; boundary_oversampling=2.5)))
    dist, w1 = zeros(nf), zeros(nf)
    for f in 1:nf
        c1, c2 = fc[1, f], fc[2, f]
        (c1 == 0 || c2 == 0) && continue
        n = geo.normal[f]
        d1 = dot(n, geo.face_centroid[f] - geo.cell_centroid[c1])
        d2 = dot(n, geo.cell_centroid[c2] - geo.face_centroid[f])
        dist[f], w1[f] = d1 + d2, d2 / (d1 + d2)
    end
    # the reference state at the centroids, the initial flow from the stream function ψ = U₀ ∫_zs^z ρ₀
    zt = range(0.0, s.H; length=20001)
    Rt = cumsum([0.0; [(ρ0(s, zt[k]) + ρ0(s, zt[k + 1])) / 2 * step(zt) for k in 1:(length(zt) - 1)]])
    Rz(z) = (t = clamp(z / step(zt), 0, length(zt) - 1.0); k = min(floor(Int, t), length(zt) - 2); Rt[k + 1] + (t - k) * (Rt[k + 2] - Rt[k + 1]))
    zd = 5000.0                       # the terrain displacement decays to zero at zd
    ψ(x) = (h = zs(s, x[1]); s.U0 * (Rz(x[3]) - Rz(h) * max(0.0, (zd - x[3]) / (zd - h))))
    Mref = vector_potential_flux(cx, x -> SVector(0.0, -ψ(x), 0.0)) ./ geo.area
    ρref = [ρ0(s, x[3]) for x in geo.cell_centroid]
    Θref = [ρ0(s, x[3]) * θ0(s, x[3]) for x in geo.cell_centroid]
    pref = pressure.(Θref)
    θin = [θ0(s, x[3]) for x in geo.face_centroid]
    τc = [sponge(s, x) for x in geo.cell_centroid]
    τf = [sponge(s, x) for x in geo.face_centroid]
    work = (θ=zeros(nc), v=[zeros(nc) for _ in 1:3], q1=zeros(nf), q2=zeros(nf), vf=[zeros(nf) for _ in 1:3],
            A=[zeros(nc) for _ in 1:3], m=[zeros(nc) for _ in 1:3], p=zeros(nc), nx=nx, nz=nz, parent=parent,
            group=group, g0=g0)
    return Model(s, cx, geo, fc, cells, lateral, R, Q, dist, w1, ρref, pref, Θref, Mref, θin, fill(s.U0, nf), zeros(nf), τc, τf, work)
end

ncells(m::Model) = length(m.geo.volume)

"Threaded y = A x with the transpose At = sparse(A') (rows as columns)."
function tmul!(y, At::SparseMatrixCSC, x)
    rv, nz = rowvals(At), nonzeros(At)
    Threads.@threads for i in eachindex(y)
        acc = 0.0
        @inbounds for k in nzrange(At, i)
            acc += nz[k] * x[rv[k]]
        end
        y[i] = acc
    end
    return y
end
initial_state(m::Model) = [copy(m.ρref); copy(m.Θref); copy(m.Mref)]

function upwind!(vf, v, m::Model, inflow::AbstractVector)
    q1, q2 = m.work.q1, m.work.q2
    tmul!(q1, m.Q[1], v)
    tmul!(q2, m.Q[2], v)
    @inbounds for f in eachindex(vf)
        c1, c2 = m.fc[1, f], m.fc[2, f]
        F = m.Mref[f]         # sign only matters at the lateral faces, where M is fixed
        if c1 > 0 && c2 > 0
            vf[f] = 0.0       # set in the flux loop from the sign of the current flux
        elseif m.lateral[f]
            into = (c1 == 0) == (F > 0)
            vf[f] = into ? inflow[f] : (c1 > 0 ? q1[f] : q2[f])
        else
            vf[f] = 0.0
        end
    end
    return q1, q2
end

function tendency!(dy, y, m::Model)
    nc, nf = ncells(m), length(m.geo.area)
    ρ, Θ, M = view(y, 1:nc), view(y, nc+1:2nc), view(y, 2nc+1:2nc+nf)
    dρ, dΘ, dM = view(dy, 1:nc), view(dy, nc+1:2nc), view(dy, 2nc+1:2nc+nf)
    V, A = m.geo.volume, m.geo.area
    θ, p = m.work.θ, m.work.p
    θ .= Θ ./ ρ
    @. p = pressure(Θ) - m.pref
    fill!(dρ, 0.0); fill!(dΘ, 0.0); fill!(dM, 0.0)
    # face value of θ, upwind, and the mass and θ fluxes
    θf = m.work.vf[1]
    q1, q2 = upwind!(θf, θ, m, m.θin)
    @inbounds for f in 1:nf
        c1, c2 = m.fc[1, f], m.fc[2, f]
        F = A[f] * M[f]
        F == 0 && continue
        θF = c1 > 0 && c2 > 0 ? (F >= 0 ? q1[f] : q2[f]) : θf[f]
        if c1 > 0; dρ[c1] -= F; dΘ[c1] -= F * θF; end
        if c2 > 0; dρ[c2] += F; dΘ[c2] += F * θF; end
    end
    # cell momentum (Perot), velocity, its upwind transport
    mc, vc, Ac = m.work.m, m.work.v, m.work.A
    for k in 1:3
        tmul!(mc[k], m.R[k], M)
        vc[k] .= mc[k] ./ ρ
        fill!(Ac[k], 0.0)
    end
    for k in 1:3
        vf = m.work.vf[k]
        q1, q2 = upwind!(vf, vc[k], m, k == 1 ? m.uin : m.zin)
        a = Ac[k]
        @inbounds for f in 1:nf
            c1, c2 = m.fc[1, f], m.fc[2, f]
            F = A[f] * M[f]
            F == 0 && continue
            vF = c1 > 0 && c2 > 0 ? (F >= 0 ? q1[f] : q2[f]) : vf[f]
            c1 > 0 && (a[c1] -= F * vF)
            c2 > 0 && (a[c2] += F * vF)
        end
        a ./= V
    end
    dρ ./= V
    dΘ ./= V
    # normal momentum: advection, pressure gradient, buoyancy of the perturbations
    @inbounds for f in 1:nf
        c1, c2 = m.fc[1, f], m.fc[2, f]
        (c1 == 0 || c2 == 0) && continue
        n = m.geo.normal[f]
        w1, w2 = m.w1[f], 1 - m.w1[f]
        adv = n[1] * (w1 * Ac[1][c1] + w2 * Ac[1][c2]) + n[2] * (w1 * Ac[2][c1] + w2 * Ac[2][c2]) +
              n[3] * (w1 * Ac[3][c1] + w2 * Ac[3][c2])
        ρp = w1 * (ρ[c1] - m.ρref[c1]) + w2 * (ρ[c2] - m.ρref[c2])
        dM[f] = adv - (p[c2] - p[c1]) / m.dist[f] - g * n[3] * ρp - m.τf[f] * (M[f] - m.Mref[f])
    end
    @. dρ -= m.τc * (ρ - m.ρref)
    @. dΘ -= m.τc * (Θ - m.Θref)
    return dy
end

"θ and w of the cells on the uncut Cartesian grid (NaN below the terrain)"
function fields(m::Model, y)
    nc = ncells(m)
    nx, nz, parent, group, g0 = m.work.nx, m.work.nz, m.work.parent, m.work.group, m.work.g0
    ρ, Θ, M = y[1:nc], y[nc+1:2nc], y[2nc+1:end]
    w = tmul!(zeros(nc), m.R[3], M) ./ ρ
    u = tmul!(zeros(nc), m.R[1], M) ./ ρ
    θ = Θ ./ ρ
    T, W, Uf = fill(NaN, nx, nz), fill(NaN, nx, nz), fill(NaN, nx, nz)
    for (k, c) in enumerate(parent)
        x = g0.cell_centroid[c]
        i, j = floor(Int, (x[1] + m.s.L) / m.s.dx) + 1, floor(Int, x[3] / m.s.dz) + 1
        T[i, j], W[i, j], Uf[i, j] = θ[group[k]], w[group[k]], u[group[k]]
    end
    return T, W, Uf
end

function run(s::Setup; hours=1.0, dt=nothing, every=600.0, out=nothing)
    m = Model(s)
    y = initial_state(m)
    dt = something(dt, 0.6 * min(s.dx, s.dz) / 350 / sqrt(2))
    nsteps = round(Int, hours * 3600 / dt)
    cache = SSPRK3Cache(y)
    @printf("%d cells (Cartesian %d × %d), %d faces, dt %.3f s, %d steps\n", ncells(m), m.work.nx, m.work.nz, length(m.geo.area), dt, nsteps)
    mass0 = sum(m.geo.volume .* y[1:ncells(m)])
    t0 = time()
    for n in 1:nsteps
        ssprk3_step!(y, dt, tendency!, cache, m)
        if n % round(Int, every / dt) == 0 || n == nsteps
            nc = ncells(m)
            w = tmul!(zeros(nc), m.R[3], y[2nc+1:end]) ./ y[1:nc]
            @printf("t %6.0f s: max|w| %.3f m/s, mass %.2e, %.0f s wall\n", n * dt, maximum(abs, w),
                    sum(m.geo.volume .* y[1:nc]) / mass0 - 1, time() - t0)
            any(isnan, y) && error("NaN")
        end
        out !== nothing && (n % round(Int, 3600 / dt) == 0 || n == nsteps) && write_fields(m, y, @sprintf("%s_%02.0fh.bin", out, n * dt / 3600))
    end
    return m, y
end

function write_fields(m::Model, y, path)
        s = m.s
        T, W, U = fields(m, y)
        open(path, "w") do io
            write(io, Int64(m.work.nx), Int64(m.work.nz), s.L, s.H, s.dx, s.dz, s.h0, s.a)
            write(io, T, W, U)
        end
end

if abspath(PROGRAM_FILE) == @__FILE__
    hours = isempty(ARGS) ? 3.0 : parse(Float64, ARGS[1])
    prefix = length(ARGS) < 2 ? "gk" : ARGS[2]
    run(Setup(L=25e3, H=20e3, dx=250.0, dz=125.0, zD=12e3, xD=17e3); hours, every=1800.0, out=prefix)
end
