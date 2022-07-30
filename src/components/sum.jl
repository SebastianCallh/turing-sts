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

function (m::Sum)(x::AbstractArray{T}, t::Integer) where T <: Real
    i = 1
    results = map(m.components) do c
        l = latent_size(c)
        result = c(x[i:i+l-1], t)
        i += l
        result
    end

    eff = mapreduce(r -> r[1], +, results)
    x′ = mapreduce(r -> r[2], vcat, results)
    ϵ = mapreduce(r -> r[3], vcat, results)
    eff, x′, ϵ
end
