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
season_effects = [0.4, 0.6, 0.3, 0, -0.2, -0.5, -0.3]
x₀ = vcat(loc_lin_init, season_effects)
num_seasons = length(season_effects)
season_length = 1
num_occurences = 4
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=num_occurences)
T = length(seasons)
σ = 0.1
tt = collect(1:T)
num_test = 20

level_drift_scale = 0.2
slope_drift_scale = 0.2
season_drift_scale = 0.2
sts = LocalLinear(level_drift_scale, slope_drift_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
y = simulate(sts, T + num_test, x₀, σ)
y_test = y[:,T+1:end]
y = y[:,1:T]
scatter(tt, y', color=seasons, label=nothing, title="Data")

##

function kalman_predict(x, P, H, F, Q, R)
    x = F*x
    P = F*P*F' + Q
    y = H*x
    S = H*P*H' + R
    x, P, y, S
end

function kalman_update(x, P, r, S, H, R)
    K = P*H'/S
    x = x + K*r
    P = (I - K*H)*P
    S = H*P*H' + R
    y = H*x
    x, P, y, S
end

@model function kalman(sts, ys; forecast=0)
    obs_dim, T = size(y)
    latent_dim = latent_size(sts)
 
    x₀ ~ MvNormal(zeros(latent_dim), I)
    σ ~ Gamma(2, 0.1)
    ϵ ~ Gamma(2, 0.1)

    P = ϵ .* I(latent_dim)
    R = Diagonal(repeat([σ^2], obs_dim))
    T′ = T + forecast
    x = x₀
 
    res = map(1:T′) do t
        H, F, Q = sts(t)
        x, P, y, S = kalman_predict(x, P, H, F, Q, R)
        
        if t <= T && !ismissing(ys[t])
            r = ys[:,t] - y
            x, P, y, S = kalman_update(x, P, r, S, H, R)            
            loglik = - only(0.5 * logdet(S) + r'/S*r)
            Turing.@addlogprob! loglik
        end
        x, P, y, S
    end
 
    return (
        x=[r[1] for r in res],
        Σx=[r[2] for r in res],    
        y=[r[3] for r in res],
        Σy=[r[4] for r in res]
    )
end

function predictive(model, chain)
    gq = Turing.generated_quantities(model, Turing.MCMCChains.get_sections(chain, :parameters))
    μ = mapreduce(gq -> only.(gq.y), hcat, gq)    
    Σ = mapreduce(gq -> gq.Σy, hcat, gq)
    μ, Σ
end

level_scale, slope_scale = 0.5, 0.5
sts = LocalLinear(level_scale, slope_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
prior_model = kalman(sts, similar(y, Missing))
prior_chain = sample(prior_model, Prior(), 100)
prior_μ, prior_Σ = predictive(prior_model, prior_chain)

prior_plt = plot(tt, prior_μ, label=nothing, color=1, title="Prior predictive")
scatter!(prior_plt, tt, y', color=seasons, label="Data")

##
model = kalman(sts, y)
chain = sample(model, NUTS(), 1000)
post_μ, post_Σ = predictive(model, chain)
mean(group(chain, :x₀))
x₀
@assert isapprox(mean(chain, "x₀[1]"), x₀[1]; atol=0.2)
@assert isapprox(mean(chain, "x₀[2]"), x₀[2]; atol=0.2)
@assert isapprox(mean(chain, "x₀[3]"), x₀[3]; atol=0.2)
@assert isapprox(mean(chain, "x₀[4]"), x₀[4]; atol=0.2)
@assert isapprox(mean(chain, "x₀[5]"), x₀[5]; atol=0.2)
@assert isapprox(mean(chain, "x₀[6]"), x₀[6]; atol=0.2)
@assert isapprox(mean(chain, :σ), σ; atol=0.1)

## 
mean_conf(Σ) = 2 .* mean(sqrt.(only.(diag.(Σ))); dims=2)

posterior_plt = plot(tt, mean(post_μ; dims=2), ribbon=mean_conf(post_Σ), label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y', label="Data", color=seasons)

## 
steps = num_test
forecast_μ, forecast_Σ = predictive(kalman(sts, y; forecast=steps), chain)
forecast_plt = plot(1:length(y)+steps, mean(forecast_μ; dims=2), ribbon=mean_conf(forecast_Σ), label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y', label="Train data", color=seasons)
scatter!(forecast_plt, tt[end]+1:tt[end]+num_test, y_test', label="Test data", color="gray")