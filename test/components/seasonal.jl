@testset "seasonal" begin
    drift_scale = 0.1
    num_seasons = 4
    season_length = 2
    m = Seasonal(num_seasons, season_length, drift_scale)
    x = collect(1:4)
    
    y1, x1, ϵ1 = m(x, 1)
    @test all(ϵ1 .== 0) # no new season => no drift scale

    y2, x2, ϵ2 = m(x, season_length)
    @test only(y2) == x[1] # observe first effect
    @test all(x2 .== x[[2, 3, 4, 1]]) # cycle seasons
    @test all(ϵ2 .== vcat(drift_scale, zeros(num_seasons-1))) # new season => drift scale    
end