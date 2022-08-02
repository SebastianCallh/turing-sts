function predictive(model, chain)
    gq = Turing.generated_quantities(model, Turing.MCMCChains.get_sections(chain, :parameters))
    μ = mapreduce(gq -> gq.y, hcat, gq)    
    Σ = mapreduce(gq -> gq.Σy, hcat, gq)

    # 1D hack
    μ = only.(μ)
    Σ = only.(Σ)

    μ, Σ
end