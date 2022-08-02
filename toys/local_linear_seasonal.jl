using StatsPlots
using Turing
using Distributions
using LinearAlgebra
using Statistics
using Random

using TuringSTS

##
Random.seed!(12345)

loc_lin_init = [-0.8, 0.2]
season_effects = [7.5, 5, -2, -5]
x₀ = vcat(loc_lin_init, season_effects)
num_seasons = length(season_effects)
season_length = 5
num_occurences = 4
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=num_occurences)
T = length(seasons)
season_colors = isempty(seasons) ? 1 : seasons
σ = 0.1
tt = collect(1:T)
num_test = 10

level_drift_scale = 2
slope_drift_scale = 0.2
season_drift_scale = 0.2
sts = LocalLinear(level_drift_scale, slope_drift_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
x, y = simulate(sts, T + num_test, x₀, σ)
y_test = y[:,T+1:end]
y = y[:,1:T]

scatter(tt, y', color=season_colors, label=nothing, title="Data")

##
prior_model = kalman(sts; forecast=length(y))
prior_chain = sample(prior_model, Prior(), 100)
prior_μ, prior_Σ = predictive(prior_model, prior_chain)

prior_plt = plot(tt, prior_μ, label=nothing, color=1, title="Prior predictive")
scatter!(prior_plt, tt, y', color=season_colors, label="Data")

##
Random.seed!(12354)
model = kalman(sts, y)
chain = sample(model, NUTS(), 1000)
post_μ, post_Σ = predictive(model, chain)

##
posterior_plt = plot(tt, mean(post_μ; dims=2), ribbon=mean(post_Σ), label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y', label="Data", color=season_colors)

## 
steps = num_test
forecast_μ, forecast_Σ = predictive(kalman(sts, y; forecast=steps), chain)
forecast_plt = plot(1:length(y)+steps, mean(forecast_μ; dims=2), ribbon=mean(forecast_Σ; dims=2), label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y', label="Train data", color=season_colors)
scatter!(forecast_plt, tt[end]+1:tt[end]+num_test, y_test', label="Test data", color="gray")