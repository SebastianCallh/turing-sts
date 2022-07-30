using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization
using Random
using TuringSTS

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

##
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
        new_season = mod(t, season_length) == 1 && 1 < t
        seasonal_trans_ = new_season ? seasonal_trans : diagm(ones(4))
        seasonal_drift_scale_ = repeat([new_season ? season_drift_scale : 0], num_seasons)
        trans = blockdiagonal([local_lin_trans, seasonal_trans_])
        x_drift_scale = vcat(loclin_drift_scale, seasonal_drift_scale_)

        x = trans * x + x_drift_scale*randn()
        y = obs * x .+ σ*randn()
    end
end

function generate(tt, x, sts, σ)
    mapreduce(vcat, tt) do t
        y, x′, ϵ = sts(x, t)
        x = x′ .+ ϵ.*randn(size(x))
        y .+ σ.*randn()
    end
end

T = 24
σ = 0.1
tt = collect(1:T)
loc_lin_init = [0.8, -0.05]
season_effects = [1, 2, 0, -2]
x₀ = vcat(loc_lin_init, season_effects)
num_seasons = length(season_effects)
season_length = 4
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=num_seasons)[1:T]
level_drift_scale = 0.1
slope_drift_scale = 0.05
season_drift_scale = 0.1
sts = LocalLinear(level_drift_scale, slope_drift_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
y = generate(tt, x₀, sts, σ)
scatter(tt, y, color=seasons, label=nothing, title="Data")

##
@model function sts_model(T, sts; forecast=0)
    D = latent_size(sts)
    x₀ ~ MvNormal(zeros(D), I)
    z ~ filldist(Normal(0, 1), D, T)
    σ ~ truncated(Normal(0, 0.1); lower=0)
    
    x = x₀
    T′ = T + forecast
    μ = map(1:T′) do t
        y, x′, ϵ = sts(x, t)
        x = x′ .+ ϵ .* z[:,min(T, t)]
        only(y)
    end

    Σ = I*σ^2
    y ~ MvNormal(μ, Σ)
end

level_scale, slope_scale = 0.1, 0.1
sts = LocalLinear(level_scale, slope_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
model = sts_model(T, sts)
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

## 
posterior_plt = plot(tt, post_predictive', color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y, label="Data", color=seasons)

steps = 10
y_forecast = Array(predict(sts_model(T, sts; forecast=steps), chain));
forecast_plt = plot(1:length(y)+steps, y_forecast', color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y, label="Data", color=seasons)