"""
    euler_forward_step!(C_next, C_curr, dt, rhs_cache, rhs_callback, args...)

Führt einen allokationsfreien expliziten Euler-Vorwärts-Schritt aus:
C_next = C_curr + dt * RHS.
Nutzt vorallokierten Speicher und optimierte Vektorisierung über @inbounds.
"""
function euler_forward_step!(
    C_next::Vector{Float64}, 
    C_curr::Vector{Float64}, 
    dt::Float64, 
    rhs_cache::Vector{Float64}, 
    rhs_callback::Function, 
    args...
)
    # Spatial Operator aufrufen (schreibt in rhs_cache)
    rhs_callback(rhs_cache, C_curr, args...)
    
    # In-place Vektor-Update ohne Grenzenüberprüfung
    @inbounds for i in eachindex(C_next)
        C_next[i] = C_curr[i] + dt * rhs_cache[i]
    end
    return C_next
end
