using Test
using Plots
include("../../src/core/datastructs.jl")
include("../../src/clustering/spectral.jl")
include("../../src/clustering/laplacian.jl")
include("../../src/clustering/lossfun.jl")
include("../../src/utils/derivative/derivative_checker.jl")

x = [1 2.9 2; 3.0 4.5 4; 1.8 4.5 3]
k = 2
d = 3
ntrain = 2
mytrain = trainingData(x, [1, 2, 2], ntrain)

f = θ -> loss_fun(x, k, d, θ, mytrain)[1]
df = θ -> loss_fun(x, k, d, θ, mytrain)[2]

f([1.0, 2.0, 3.0])
df([1.0, 2.0, 3.0])

(r1, r2, r3, r4) = checkDerivative(f, df, [0.1, 0.1, 0.2], nothing, 1, 14)
r3
r4

@test r4[1] ≈ 2 atol = 0.5
