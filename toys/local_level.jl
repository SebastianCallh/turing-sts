using StatsPlots
using Turing
using Distributions
using LinearAlgebra
using Statistics
using Random

using TuringSTS

##
Random.seed!(12345)

local_level_init = [-0.8]
x₀ = vcat(local_level_init)
T = 5
σ = 0.01
tt = collect(1:T)
num_test = 5

level_drift_scale = 0.1
sts = LocalLevel(level_drift_scale)

y, _ = simulate(sts, T + num_test, x₀, σ)
y_test = y[:,T+1:end]
y = y[:,1:T]
tt_test = tt[end]+1:tt[end]+num_test

data_plt = scatter(tt, y', label=nothing, color=1, title="Train data")
scatter!(data_plt, tt_test, y_test', label=nothing, color=2, title="Test data")

##
prior_model = kalman(sts, y)
prior_chain = sample(prior_model, Prior(), 100)
prior_μ, prior_Σ = predictive(prior_model, prior_chain)
prior_plt = plot(tt, mean(prior_μ; dims=2), ribbon=mean(prior_Σ; dims=2), label=nothing, title="Prior predictive")
scatter!(prior_plt, tt, y', label="Data")

##
Random.seed!(1)
model = kalman(sts, y)
chain = sample(model, NUTS(), 1000)
post_μ, post_Σ = predictive(model, chain)
@assert isapprox(mean(chain, "x₀[1]"), x₀[1]; atol=0.25)

## 
post_μ, post_Σ = predictive(model, chain)
post_plt = plot(tt, mean(post_μ; dims=2), ribbon=mean(post_Σ; dims=2), label=nothing, title="Prior predictive")
scatter!(post_plt, tt, y', label="Data")

## 
steps = num_test
forecast_μ, forecast_Σ = predictive(kalman(sts, y; forecast=steps), chain)
forecast_plt = plot(1:length(y)+steps, mean(forecast_μ; dims=2), ribbon=mean(forecast_Σ; dims=2), label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y', label="Train data")
scatter!(forecast_plt, tt[end]+1:tt[end]+num_test, y_test', label="Test data", color="gray")