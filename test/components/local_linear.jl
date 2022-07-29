@testset "local_linear" begin
    m = LocalLinear()
    x = [1, 2]
    
    y₁, x₁ = m(x, 1)
    @test only(y₁) == x[1] # observe level
    @test all(x₁ .== [3, 2])
end