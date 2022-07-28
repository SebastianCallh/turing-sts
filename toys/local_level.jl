using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization
using Random

Random.seed!(1234)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function true_fn(t, α, ϵ, σ)
    map(t) do _
        y = α + σ*randn()
        α += ϵ*randn()    
        y  
    end
end

N = 50
σ = 0.1
t = collect(1:N)
α₀ = 0.3
ϵ = 0.1
y = Float32.(true_fn(t, α₀, ϵ, σ))
scatter(t, y)

@model function local_level(T, type, forecast=0)
    ϵ ~ Gamma(2, .1)
    α_z ~ MvNormal(zeros(T-1), I)
    α₀ ~ Normal(0, 1)
    σ ~ Exponential(0.1)

    T′ = T + forecast
    α = Vector{type}(undef, T′)
    α[1] = α₀
    for t in 2:T′
        α[t] = α[t-1] + ϵ*α_z[min(T-1, t-1)]
    end

    μ = α
    Σ = I*σ^2
    y ~ MvNormal(μ, Σ)

    return (;α, ϵ, σ)
end

model = local_level(N, eltype(y))
prior_samples = mapreduce(i -> rand(model).y, hcat, 1:50)
prior_plt = plot(t, prior_samples, label=nothing, color=1, title="Prior predictive check")
scatter!(prior_plt, t, y, color=2, label="Data")

chain = sample(model | (;y), NUTS(), 1000)
posterior_samples = predict(model, chain)
@assert isapprox(mean(chain, :σ), σ; atol=0.1)
@assert isapprox(mean(chain, :ϵ), ϵ; atol=0.1)
@assert isapprox(mean(chain, :α₀), α₀; atol=0.1)

posterior_plt = plot(t, Array(posterior_samples)', color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, t, y, label="Data", color=2)

steps = 10
y_forecast = Array(group(predict(local_level(N, eltype(y), steps), chain), :y))
forecast_plt = plot(1:length(y)+steps, y_forecast', color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, t, y, label="Data", color=2)