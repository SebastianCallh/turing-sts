using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization
using Random

Random.seed!(1234)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function true_fn(t, α, β, αϵ, βϵ, σ)
    map(t) do _
        y = α + σ*randn()
        α += β + αϵ*randn()
        β += βϵ*randn()
        Float32(y)
    end
end

N = 50
σ = 0.1
t = collect(1:N)
α₀ = 1.3
β₀ = -0.05
αϵ = 0.1
βϵ = 0.05
y = true_fn(t, α₀, β₀, αϵ, βϵ, σ)
scatter(t, y)

@model function local_linear(T, type, forecast=0)
    αϵ ~ Gamma(2, 1e-2)
    βϵ ~ Gamma(2, 1e-2)
    α_z ~ MvNormal(zeros(T-1), I)
    β_z ~ MvNormal(zeros(T-1), I)
    α₀ ~ Normal(0, 1)
    β₀ ~ Normal(0, 0.1)
    σ ~ Exponential(0.1)

    T′ = T + forecast
    α = Vector{type}(undef, T′)
    β = Vector{type}(undef, T′)
    α[1] = α₀
    β[1] = β₀
    for t in 2:T′
        t_clamped = min(T-1, t-1)
        α[t] = α[t-1] + β[t-1] + αϵ*α_z[t_clamped]
        β[t] = β[t-1] + βϵ*β_z[t_clamped]
    end

    μ = α
    Σ = I*σ^2
    y ~ MvNormal(μ, Σ)
end

model = local_linear(N, eltype(y))
prior_samples = mapreduce(i -> rand(model).y, hcat, 1:50)
prior_plt = plot(t, prior_samples, label=nothing, color=1, title="Prior predictive check")
scatter!(prior_plt, t, y, color=2, label="Data")

chain = sample(model | (;y), NUTS(), 2000)
posterior_samples = predict(model, chain)
@assert isapprox(mean(chain, :σ), σ; atol=0.05)
@assert isapprox(mean(chain, :αϵ), αϵ; atol=0.5)
@assert isapprox(mean(chain, :βϵ), βϵ; atol=0.01)
@assert isapprox(mean(chain, :α₀), α₀; atol=0.2)
@assert isapprox(mean(chain, :β₀), β₀; atol=0.1)

posterior_plt = plot(t, Array(posterior_samples)', color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, t, y, label="Data", color=2)

steps = 10
y_forecast = Array(predict(local_linear(N, eltype(y), steps), chain));
forecast_plt = plot(1:length(y)+steps, y_forecast', color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, t, y, label="Data", color=2)