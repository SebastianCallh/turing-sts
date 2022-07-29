@testset "seasonal" begin
    drift_scale = 0.1
    num_seasons = 4
    season_length = 2
    m = Seasonal(num_seasons, season_length, drift_scale)
    x = collect(1:4)
    
    y₁, x₁, ϵ₁ = m(x, 1)
    @test only(y₁) == x[1] # observe first effect
    @test all(x₁ .== x[[2, 3, 4, 1]]) # cycle seasons
    @test all(ϵ₁ .== vcat(drift_scale, zeros(num_seasons-1))) # new season => drift scale
    
    y₂, x₂, ϵ₂ = m(x, 2)
    @test all(ϵ₂ .== 0) # no new season => no drift scale
end