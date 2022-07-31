abstract type Component end

"""
    simulate(model::Component, T, x, σ)

Simulates from the provided `model` for `T` time steps
with initial state `x` and observation noise `σ`.
"""
function simulate(model::Component, T, x, σ)
    mapreduce(hcat, 1:T) do t
        obs, trans, Σx = model(t)
        x = rand(MvNormal(trans*x, Σx))

        μy = obs*x
        Σy = σ^2 .* I(size(μy, 1))
        y = rand(MvNormal(μy, Σy))
    end
end