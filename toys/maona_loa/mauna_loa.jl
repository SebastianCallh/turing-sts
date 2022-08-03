using DataFrames, CSV, StatsPlots, Turing, TuringSTS, STSlib

##
co2_path = "co2.csv"
isfile(co2_path) || download("https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv", co2_path)

df = CSV.read(co2_path, DataFrame, skipto=58, missingstring="-99.99") |>
    x -> rename(x[:,[1,2,4,5]], ["Year", "Month", "Date", "Co2"]) |>
    x -> sort(x, :Date)

train_start = 12 * 8
train_end = train_start + 12*3
test_end = train_end + 12*1
df_test = df[train_end+1:test_end,:]
df = df[train_start:train_end,:]

data_plt = @df df scatter(:Date, :Co2)
@df df_test scatter!(data_plt, :Date, :Co2)
##

seasons = df.Month
num_seasons = 12
season_length = 1
drift_scale = 0.1
y0 = first(df[!, :Co2])
y = Matrix(df[!, :Co2]' .- y0)
y_test = Matrix(df_test[!, :Co2]' .- y0)
tt = 1:size(y, 2)
tt_test = tt[end]+1:tt[end]+size(y_test, 2)
sts = LocalLinear(0.1, 0.1) + Seasonal(num_seasons, season_length, drift_scale)
prior_model = kalman(sts; forecast=length(y))
prior_chain = sample(prior_model, Prior(), 100)
prior_μ, prior_Σ = predictive(prior_model, prior_chain)

prior_plt = plot(tt, prior_μ, label=nothing, color=1, title="Prior predictive")
scatter!(prior_plt, tt, y', color=seasons, label="Data")

##
model = kalman(sts, y)
chain = sample(model, NUTS(), 1000)

##
post_μ, post_Σ = predictive(model, chain)
posterior_plt = plot(tt, mean(post_μ; dims=2), ribbon=mean(post_Σ), label=nothing, title="Mauna Loa posterior predictive check")
scatter!(posterior_plt, tt, y', label="Data", color=seasons)
savefig(posterior_plt, "posterior.png")

## 
steps = length(tt_test)
forecast_μ, forecast_Σ = predictive(kalman(sts, y; forecast=steps), chain)
forecast_plt = plot(1:length(y)+steps, mean(forecast_μ; dims=2), ribbon=mean(forecast_Σ; dims=2), label=nothing, title="Mauna Loa posterior forecast")
scatter!(forecast_plt, tt, y', label="Train data")
scatter!(forecast_plt, tt_test, y_test', label="Test data", color="gray")
savefig(forecast_plt, "forecast.png")