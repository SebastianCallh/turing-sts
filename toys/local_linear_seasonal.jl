using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics
using ReverseDiff, Memoization, BlockDiagonals
using Random

Random.seed!(1234)
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

function true_fn(t, x₀, x_drift_scale, season_length, σ)
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

    mapreduce(vcat, t) do t
        seasonal_trans_ = mod(t-1, season_length) == 0 ? seasonal_trans : diagm(ones(4))
        trans = BlockDiagonal([local_lin_trans, seasonal_trans_])
        x = trans * x + x_drift_scale*randn()
        y = obs * x .+ σ*randn()
    end
end

N = 80
σ = 0.1
t = collect(1:N)
loc_lin_init = [1.3, -0.05]
season_effects = [1, 2, 0, -2]
x₀ = vcat(loc_lin_init, season_effects)
num_seasons = length(season_effects)
season_length = 4
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=div(T, num_seasons * season_length))
x_drift_scale = vcat(0.1, 0.05, repeat([0.33], num_seasons))

y = true_fn(t, x₀, x_drift_scale, season_length, σ)
scatter(t, y, color=seasons)
