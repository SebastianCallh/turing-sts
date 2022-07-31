struct LocalLinear <: Component
    obs::Matrix{Int64}
    trans::Matrix{Int64}
    trans_cov::Diagonal{Float64, Vector{Float64}}
end

"""

    LocalLinear(level_scale::Real, slope_scale::Real)

"""
LocalLinear(level_scale=1., slope_scale=1.) = LocalLinear([1 0], [1 1; 0 1], Diagonal([level_scale^2, slope_scale^2]))
latent_size(m::LocalLinear) = size(m.obs, 2)
observed_size(m::LocalLinear) = size(m.obs, 1)

function (m::LocalLinear)(t::Integer)
    m.obs, m.trans, m.trans_cov
end
