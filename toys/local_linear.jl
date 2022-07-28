using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization
using Random

Random.seed!(1234)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function true_fn(t, x₀, x_scale, σ)
    trans = [1 1 ; 0 1]
    obs = [1 0]
    x = x₀
    mapreduce(vcat, t) do _
        x = trans * x + x_scale*randn()
        y = obs * x .+ σ*randn()
    end
end

T = 50
σ = 0.1
tt = collect(1:T)
x₀ = [1.3, -0.05]
x_scale = [0.1, 0.05]
y = true_fn(tt, x₀, x_scale, σ)
scatter(tt, y)

@model function local_linear(T; obs_noise_scale=1., level_scale=1., slope_scale=1., forecast=0)
    trans = [1 1; 0 1]
    obs = [1 0]
    D = size(trans, 1)

    x₀ ~ MvNormal(zeros(D), I)
    x_noise ~ filldist(Normal(0, 1), D, T)
    ϵ ~ MvNormal(zeros(D), diagm([level_scale, slope_scale]))
    σ ~ truncated(Normal(0, obs_noise_scale); lower=0)

    x = x₀
    T′ = T + forecast
    μ = map(1:T′) do t
        x = trans * x .+ ϵ .* x_noise[:,min(T, t)]
        only(obs * x)
    end

    Σ = I*σ^2
    y ~ MvNormal(μ, Σ)
end

model = local_linear(T; level_scale=0.1, slope_scale=0.1)
prior_samples = mapreduce(i -> rand(model).y, hcat, 1:50)
prior_plt = plot(tt, prior_samples, label=nothing, color=1, title="Prior predictive check")
scatter!(prior_plt, tt, y, color=2, label="Data")

chain = sample(model | (;y), NUTS(), 1000)
posterior_samples = predict(model, chain)
@assert isapprox(mean(chain, :σ), σ; atol=0.05)
@assert isapprox(mean(chain, "x₀[1]"), x₀[1]; atol=0.5)
@assert isapprox(mean(chain, "x₀[2]"), x₀[2]; atol=0.5)

posterior_plt = plot(tt, Array(posterior_samples)', color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y, label="Data", color=2)

steps = 10
y_forecast = Array(predict(local_linear(T; forecast=steps), chain));
forecast_plt = plot(1:length(y)+steps, y_forecast', color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y, label="Data", color=2)