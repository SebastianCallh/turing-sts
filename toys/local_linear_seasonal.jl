using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization
using Random
using BlockDiagonals

Random.seed!(1234)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function blockdiagonal(mats::Vector{Matrix{T}}) where T
    M, N = mapreduce(size, .+, mats)
    res = zeros(T, M, N)
    cur_ind = CartesianIndex(0, 0)
    for m in mats
        m_inds = CartesianIndices(m)
        for i in m_inds
            res[i + cur_ind] = m[i]
        end
        cur_ind += m_inds[end]
    end
    res
end


function true_fn(t, x₀, level_drift_scale, slope_drift_scale, season_drift_scale, season_length, num_seasons, σ)
    local_lin_trans = [
        1 1;
        0 1;
    ]
    seasonal_trans = [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1;
        1 0 0 0;
    ]

    obs = [1 0 1 0 0 0]
    x = x₀

    loclin_drift_scale = vcat(level_drift_scale, slope_drift_scale)
    mapreduce(vcat, t) do t
        new_season = mod(t-1, season_length) == 0
        seasonal_trans_ = new_season ? seasonal_trans : diagm(ones(4))
        seasonal_drift_scale_ = repeat([new_season ? season_drift_scale : 0], num_seasons)
        trans = blockdiagonal([local_lin_trans, seasonal_trans_])
        x_drift_scale = vcat(loclin_drift_scale, seasonal_drift_scale_)

        x = trans * x + x_drift_scale*randn()
        y = obs * x .+ σ*randn()
    end
end

T = 80
σ = 0.1
tt = collect(1:T)
loc_lin_init = [1.3, -0.05]
season_effects = [1, 2, 0, -2]
x₀ = vcat(loc_lin_init, season_effects)
num_seasons = length(season_effects)
season_length = 4
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=div(T, num_seasons * season_length))
level_drift_scale = 0.1
slope_drift_scale = 0.05
season_drift_scale = 0.1
y = true_fn(tt, x₀, level_drift_scale, slope_drift_scale, season_drift_scale, season_length, num_seasons, σ)
scatter(tt, y, color=seasons)

@model function local_linear_seasonal(
    T, num_seasons;
    obs_noise_scale=1.,
    level_scale=1.,
    slope_scale=1.,
    season_length=1,
    season_drift_scale=1.,
    forecast=0,
)
    obs = [1 0 1 0 0 0]
    loclin_trans = [
        1 1;
        0 1
    ]
    loclin_scales = vcat(level_scale, slope_scale)
    seasonal_trans = [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1;
        1 0 0 0;
    ]
    D = size(obs, 2)
    season_eltype = eltype(season_drift_scale)

    x₀ ~ MvNormal(zeros(D), I)
    x_trans_noise ~ filldist(Normal(0, 1), D, T)
    σ ~ truncated(Normal(0, obs_noise_scale); lower=0)
    
    x = x₀
    T′ = T + forecast
    μ = map(1:T′) do t
        new_season = mod(t-1, season_length) == 0
        ses_trans = new_season ? seasonal_trans : diagm(ones(eltype(seasonal_trans), 4))
        ses_scale = new_season ? vcat(season_drift_scale, zeros(season_eltype, num_seasons-1)) : zeros(season_eltype, num_seasons) 

        trans = blockdiagonal([loclin_trans, ses_trans])
        x_trans_scale = vcat(loclin_scales, ses_scale)
        x = trans * x .+ x_trans_scale .* x_trans_noise[:,min(T, t)]
        only(obs * x)
    end

    Σ = I*σ^2
    y ~ MvNormal(μ, Σ)
end

model = local_linear_seasonal(T, num_seasons; level_scale=0.05, slope_scale=0.05, season_drift_scale=0.1, season_length, obs_noise_scale=0.1)
prior_predictive = mapreduce(i -> rand(model).y, hcat, 1:50)
prior_plt = plot(tt, prior_predictive, label=nothing, color=1, title="Prior predictive check")
scatter!(prior_plt, tt, y, color=seasons, label="Data")

chain = sample(model | (; y), NUTS(), 1000)
post_predictive = Array(predict(model, chain))
@assert isapprox(mean(chain, "x₀[1]"), x₀[1]; atol=0.5)
@assert isapprox(mean(chain, "x₀[2]"), x₀[2]; atol=0.5)
@assert isapprox(mean(chain, "x₀[3]"), x₀[3]; atol=0.5)
@assert isapprox(mean(chain, "x₀[4]"), x₀[4]; atol=0.5)
@assert isapprox(mean(chain, "x₀[5]"), x₀[5]; atol=0.5)
@assert isapprox(mean(chain, "x₀[6]"), x₀[6]; atol=0.5)
@assert isapprox(mean(chain, :σ), σ; atol=0.05)

posterior_plt = plot(tt, post_predictive', color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y, label="Data", color=seasons)

steps = 15
y_forecast = Array(predict(local_linear_seasonal(T, num_seasons; level_scale=0.05, slope_scale=0.05, season_drift_scale=0.1, season_length, obs_noise_scale=0.1, forecast=steps), chain));
forecast_plt = plot(1:length(y)+steps, y_forecast', color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y, label="Data", color=seasons)