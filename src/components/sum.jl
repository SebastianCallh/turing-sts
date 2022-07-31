"""

Sum(components::Vector{Component}, obs_noise_scale::Real)

"""
struct Sum <: Component
    components::Vector{Component}
end

latent_size(m::Sum) = mapreduce(latent_size, +, m.components)
observed_size(m::Sum) = mapreduce(observed_size, +, m.components)
Base.length(m::Sum) = length(m.components)
Base.:(+)(c1::Component, c2::Component) = Sum(vcat(c1, c2))

function (m::Sum)(t::Integer)
    results = map(c -> c(t), m.components)
    obs = mapreduce(r -> r[1], hcat, results)
    trans = blockdiagonal([r[2] for r in results])
    trans_cov = blockdiagonal([r[3] for r in results])
    obs, trans, trans_cov
end
