"""
Finite elements of arbitrary order on triangles and quadrilaterals: Lagrange,
discontinuous, Raviart–Thomas and Nédélec families, reference elements built
from a Legendre prime basis, function spaces on a `Plex`, and assembly.
"""
module FEM

using LinearAlgebra
using SparseArrays
using StaticArrays
using ..Topology
using ..Geometry

export ElementFamily, Lagrange, Discontinuous, RaviartThomas, Nedelec, function_space, value_size,
    field, lagrange, discontinuous, raviart_thomas, nedelec
export QuadratureRule, quadrature_rule, gauss_legendre, gauss_lobatto, legendre
export ReferenceElement, DofFunctional, num_dofs, tabulate
export FESpace, reference_element, cell_dofs, evaluate, Tabulation, cell_basis, precompute_basis
export assemble_mass, assemble_bilinear, assemble_divergence, assemble_coriolis, assemble_load, project

include("Polynomials.jl")
include("Quadrature.jl")
include("Families.jl")
include("ReferenceElement.jl")
include("Lagrange.jl")
include("EdgeElements.jl")
include("FESpace.jl")
include("Assembly.jl")

end # module
