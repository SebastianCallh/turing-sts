using StatsPlots, Turing, Distributions, LinearAlgebra, Statistics, Random
using TuringSTS

##
Random.seed!(12345)

loc_lin_init = [-0.8, 0.4]
season_effects = [0.1, 0.2, 0, -0.2]
x₀ = vcat(loc_lin_init, season_effects)
num_seasons = length(season_effects)
season_length = 3
num_occurences = 2
seasons = repeat(collect(1:num_seasons), inner=season_length, outer=num_occurences)
T = length(seasons)
σ = 0.1
tt = collect(1:T)
num_test = 20

level_drift_scale = 0.1
slope_drift_scale = 0.1
season_drift_scale = 0.1
sts = LocalLinear(level_drift_scale, slope_drift_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
y = simulate(sts, T + num_test, x₀, σ)
y_test = y[:,T+1:end]
y = y[:,1:T]
scatter(tt, y', color=seasons, label=nothing, title="Data")

##
@model function kalman(sts, y; forecast=0)
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

        x = F*x
        P = F*P*F' + Q
        ŷ = H*x
        S = H*P*H' + R

        if t <= T && !ismissing(y[t])
            v = y[:,t] - ŷ
            K = P*H'*inv(S)
            x = x + K*v            
            P = (I - K*H)*P

            # corrected observation
            S = H*P*H' + R
            ŷ = H*x

            C = cholesky(S)
            loglik = - only(0.5 * logdet(S) + v'*(C \ v))
            Turing.@addlogprob! loglik
        end
        ŷ, S, x, P
    end
 
    return (
        y=[r[1] for r in res],
        Σy=[r[2] for r in res],
        x=[r[3] for r in res],
        Σx=[r[4] for r in res],
    )
end

function predictive(model, chain)
    gq = Turing.generated_quantities(model, Turing.MCMCChains.get_sections(chain, :parameters))
    mapreduce(gq -> only.(gq.y), hcat, gq)
end

level_scale, slope_scale = 0.5, 0.5
sts = LocalLinear(level_scale, slope_scale) + Seasonal(num_seasons, season_length, season_drift_scale)
prior_model = kalman(sts, similar(y, Missing))
prior_chain = sample(prior_model, Prior(), 100)
prior_y = predictive(prior_model, prior_chain)
prior_plt = plot(tt, prior_y, label=nothing, color=1, title="Prior predictive")
scatter!(prior_plt, tt, y', color=seasons, label="Data")

##
model = kalman(sts, y)
chain = sample(model, NUTS(), 1000)
post_gq = Turing.generated_quantities(model, Turing.MCMCChains.get_sections(chain, :parameters))
post_y = predictive(model, chain)
@assert isapprox(mean(chain, "x₀[1]"), x₀[1]; atol=0.2)
@assert isapprox(mean(chain, "x₀[2]"), x₀[2]; atol=0.2)
@assert isapprox(mean(chain, "x₀[3]"), x₀[3]; atol=0.2)
@assert isapprox(mean(chain, "x₀[4]"), x₀[4]; atol=0.2)
@assert isapprox(mean(chain, "x₀[5]"), x₀[5]; atol=0.2)
@assert isapprox(mean(chain, "x₀[6]"), x₀[6]; atol=0.2)
@assert isapprox(mean(chain, :σ), σ; atol=0.1)

## 
posterior_plt = plot(tt, post_y, color=1, alpha=0.1, label=nothing, title="Posterior predictive check")
scatter!(posterior_plt, tt, y', label="Data", color=seasons)

## 
steps = 50
forecast_y = predictive(kalman(sts, y; forecast=steps), chain)
forecast_plt = plot(1:length(y)+steps, forecast_y, color=1, alpha=0.1, label=nothing, title="Posterior forecast")
scatter!(forecast_plt, tt, y', label="Data", color=seasons)
scatter!(forecast_plt, tt[end]+1:tt[end]+num_test, y_test', label=nothing, color="gray")