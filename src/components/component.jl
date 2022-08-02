abstract type Component end

"""
    simulate(model::Component, T, x, σ)

Simulates from the provided `model` for `T` time steps
with initial state `x` and observation noise `σ`.
"""
function simulate(sts::Component, T, x, σ; jitter=1e-8)
    res = mapreduce(hcat, 1:T) do t
        H, F, Q = sts(t)
        latent_dim = size(F, 1)
        x = rand(MvNormal(F*x, Q .+ jitter .* abs.(diagm(randn(latent_dim)))))
        y = H*x
        (;x, y)
    end
    mapreduce(r -> r.x, hcat, res), mapreduce(r -> r.y, hcat, res)
end 