struct Seasonal <: Component
    obs::Matrix{Int64}
    trans::Matrix{Int64}
    trans_cov::Diagonal{Float64}
    season_length::Int64
end

"""

    Seasonal(num_seasons::Integer, season_length::Integer, drift_scale::Real)

"""
Seasonal(num_seasons::Integer, season_length::Integer, drift_scale::Real) = begin
    obs = Matrix(vcat(1, zeros(Int64, num_seasons-1))')
    trans = diagm(ones(num_seasons))[:,vcat(num_seasons, 1:num_seasons-1)]
    trans_cov = Diagonal(vcat(drift_scale^2, zeros(num_seasons-1)))
    Seasonal(obs, trans, trans_cov, season_length)
end
latent_size(m::Seasonal) = size(m.obs, 2)
observed_size(m::Seasonal) = size(m.obs, 1)


function (m::Seasonal)(t::Integer)
   num_seasons = size(m.trans, 1)
   new_season = mod(t-1, m.season_length) == 0 && 1 < t   
   trans = new_season ? m.trans : diagm(ones(eltype(m.trans), num_seasons))
   trans_cov = new_season ? m.trans_cov : Diagonal(zeros(eltype(m.trans_cov), num_seasons))
   m.obs, trans, trans_cov
end
