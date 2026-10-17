include("SphericalMesh.jl")
using .SphericalMesh

# Initialize a spherical grid with Radius = 6371.0 km (Earth)
R = 6371.0
verts, cells = generate_icosahedron(R)
mesh = build_dag!(verts, cells, R)

println("--- Mesh Successfully Created ---")
println("Total Grid Points in DAG: ", mesh.num_points)
println("Vertices Count: ", length(mesh.strata[0]))
println("Edges Count:    ", length(mesh.strata[1]))
println("Cells Count:    ", length(mesh.strata[2]))

# Test geometry on the first cell
first_cell = minimum(mesh.strata[2])
cell_area = spherical_triangle_area(mesh, first_cell)
total_area = cell_area * 20 # 20 symmetric faces

println("\n--- Spherical Geometry Verification ---")
println("Single Cell Area: ", cell_area, " km^2")
println("Calculated Total Sphere Area: ", total_area, " km^2")
println("Analytical 4πR^2 Surface Area: ", 4 * pi * R^2, " km^2")

