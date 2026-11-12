# Polyhedral finite volumes: description

[polyhedral.tex](polyhedral.tex) describes the finite-volume method of POLYEDER as it is carried
over to `src/Polyhedral/`: cell complexes, the exact integration of monomials over polyhedral cells,
the geometry, and (planned) the polynomial reconstruction, the face gradient from interaction
regions and the shallow-water equations.

```sh
latexmk -pdf polyhedral.tex
```

The numbers of Table 1 (volumes of a spherical shell) come from `PolyGeometry` on
`extruded_complex(ExtrudedPlex(mesh, 1), [0.0, 0.1]; radius=1.0)`; the tests are in
`test/polyhedral.jl`.
