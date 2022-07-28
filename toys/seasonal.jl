using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization
using Random

Random.seed!(1234)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function true_fn(t, s, s_eff, σ)
    map(t) do tᵢ
        y = s_eff[s[tᵢ]] + σ * randn()
        Float32(y)
    end
end

T = 80
σ = 0.1
tt = collect(1:T)
num_seasons = 4
season_length = 4
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=div(T, num_seasons * season_length))
season_effects = [0.5, -0.5, 0.75, 0.2]
y = true_fn(tt, seasons, season_effects, σ)
scatter(tt, y, color=seasons, label=nothing, title="Data scatterplot")

@model function seasonal(T, num_seasons; season_length=1, drift_scale=1., initial_state_prior=nothing, forecast=0)
    σ ~ Exponential(0.1)
    initial_state ~ initial_state_prior !== nothing ? initial_state_prior : MvNormal(zeros(num_seasons), I)
    drift_noise ~ Normal(0, 1)

    T′ = T + forecast
    state = initial_state
    state_trans = I(num_seasons)[:, vcat(num_seasons, 1:num_seasons-1)]
    obs = vcat(1, zeros(num_seasons-1))'
    μ = map(1:T′) do t
        new_season = mod(t-1, season_length) == 0
        trans = new_season ? state_trans : I
        noise = new_season ? drift_scale .* vcat(drift_noise, zeros(num_seasons-1)) : zeros(num_seasons)
        state = trans*state + noise
        only(obs*state)
    end
    
    Σ = I * σ^2
    y ~ MvNormal(μ, Σ)
end


model = seasonal(T, num_seasons; season_length)
prior_samples = mapreduce(i -> rand(model).y, hcat, 1:50)
prior_plt = plot(tt, prior_samples, label=nothing, color=1, title="Prior predictive check")
scatter!(prior_plt, tt, y, color=2, label="Data")

chain = sample(model | (; y), NUTS(), 2000)
posterior_samples = predict(model, chain)
@assert isapprox(mean(chain, :σ), σ; atol=0.05)
mean(chain, :σ)
σ

posterior_plt = plot(tt, Array(posterior_samples)', color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y, label="Data", color=seasons)

steps = 10
y_forecast = Array(predict(seasonal(T, num_seasons; season_length, forecast=steps), chain));
forecast_plt = plot(1:length(y)+steps, y_forecast', color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y, label="Data", color=seasons)