@testset "sum" begin
    c1 = LocalLinear()
    c2 = LocalLinear()
    c = c1 + c2
    
    @test latent_size(c) == latent_size(c1) + latent_size(c2)
    @test observed_size(c) == observed_size(c1) + observed_size(c2)

    t = 1
    x = [1,2,3,4]
    y, x′, ϵ = c(x, t)
    @test all(x′ .== [3,2,7,4])
    @test only(y) == 4
end
