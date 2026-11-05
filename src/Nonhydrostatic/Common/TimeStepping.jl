# Time-stepping drivers shared by the dycores.

"""
    wicker_skamarock!(m, y, y0, ys, dt)

The three stages of the Wicker–Skamarock RK3 step (Wicker and Skamarock 2002) of
length `dt` from the base state `y0`: `ys = y0 + dt/3 T(y0)`, `y = y0 + dt/2 T(ys)`,
`y = y0 + dt T(y)`, each a HEVI `stage!(m, y, y0, ys, τ)` of the model (horizontal
terms explicit in the stage state, vertical terms implicit). `y` is the result,
`ys` work space for the stage state.
"""
function wicker_skamarock!(m, y, y0, ys, dt)
    stage!(m, ys, y0, y0, dt / 3)
    stage!(m, y, y0, ys, dt / 2)
    foreach(copyto!, ys, y)
    stage!(m, y, y0, ys, dt)
    return y
end
