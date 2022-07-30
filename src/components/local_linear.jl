struct LocalLinear <: Component
    obs::Matrix{Int64}
    trans::Matrix{Int64}
    drift_scale::Vector{Float64}
end

"""

    LocalLinear(level_scale, slope_scale)

"""
LocalLinear(level_scale=1., slope_scale=1.) = LocalLinear([1 0], [1 1; 0 1], [level_scale, slope_scale])
latent_size(m::LocalLinear) = size(m.obs, 2)
observed_size(m::LocalLinear) = size(m.obs, 1)

function (m::LocalLinear)(x::AbstractArray{T}, t::Integer) where T <: Real
    m.obs*x, m.trans*x, m.drift_scale
end
