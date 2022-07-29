"""

Sum(components::Vector{Component}, obs_noise_scale::Real)

"""
struct Sum <: Component
    components::Vector{Component}
    obs_noise_scale::Float64
end

function (m::Sum)(x::AbstractArray{T}, t::Integer) where T <: Real
    results = [c(x, t) for c in m.components]    
    obs = mapreduce(c -> c[1], hcat, results)
    trans = blockdiagonal([c[2] for c in results])
    drift_scales = mapreduce(c -> c[3], vcat, results)
    obs, trans, drift_scales
end
