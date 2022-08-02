struct LocalLevel <: Component
    obs::Matrix{Int64}
    trans::Matrix{Int64}
    trans_cov::Matrix{Float64}
end

"""

    LocalLevel(level_scale::Real)

"""
LocalLevel(level_scale=1.) = LocalLevel([1;;], [1;;], diagm([level_scale^2]))
latent_size(m::LocalLevel) = size(m.obs, 2)
observed_size(m::LocalLevel) = size(m.obs, 1)

function (m::LocalLevel)(t::Integer)
    m.obs, m.trans, m.trans_cov
end
