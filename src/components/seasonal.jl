struct Seasonal <: Component
    obs::Matrix{Int64}
    trans::Matrix{Int64}
    season_length::Int64
    drift_scale::Float64
end

"""

    Seasonal(num_seasons, season_length, drift_scale)

"""
Seasonal(num_seasons::Integer, season_length::Integer, drift_scale::Real) = begin
    obs = Matrix(vcat(1, zeros(Int64, num_seasons-1))')
    trans = diagm(ones(num_seasons))[:,vcat(num_seasons, 1:num_seasons-1)]
    Seasonal(obs, trans, season_length, drift_scale)
end
latent_size(m::Seasonal) = size(m.obs, 2)
observed_size(m::Seasonal) = size(m.obs, 1)


function (m::Seasonal)(x::AbstractArray{T}, t::Integer) where T <: Real
   num_seasons = size(m.trans, 1)
   new_season = mod(t - 1, m.season_length) == 0
   trans = new_season ? m.trans : diagm(ones(num_seasons))
   drift_scale = new_season ? vcat(m.drift_scale, zeros(num_seasons-1)) : zeros(num_seasons)
   m.obs*x, trans*x, drift_scale
end