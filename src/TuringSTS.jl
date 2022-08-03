module TuringSTS

using LinearAlgebra
using Distributions
using Turing
using STSlib

export kalman, predictive

include("inference/kalman.jl")
include("inference/utils.jl")
end
