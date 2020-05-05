using Distributions
using TensorOperations
using MLDatasets
using Random

include("comp_mat_deriv.jl")
include("../kernels/kernels.jl")
n = 20; ntest = 100
dataraw, _ = MNIST.testdata(Float64)
randseed = 1234; rng = MersenneTwister(randseed)
idx = randperm(rng, n)
X = reshape(permutedims(dataraw[:, :, idx],(3, 1, 2)), n, size(dataraw,1)*size(dataraw,2))
n, d = size(X)

fun_A(θ) = affinity_A(X, θ)[1]
deriv_A(θ) =  affinity_A(X, θ)[2]

# 1d parameter
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dAtest_fd = deriv_fd(fun_A, 1e-5)
θgrid = range(0.01, stop=2, length=ntest)
err1 = maximum(norm.(deriv_A.(θgrid) - dAtest_fd.(θgrid)))
@info "One-dim parameter test: error" err1

# multi-dimensional parameter
θgrid = rand(Uniform(0.01,2), d, ntest)
dAh = Array{Float64, 2}(undef, n, n)
err2 = 0.
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    @tensor dAh[l, j] = deriv_A(θ)[l, j, k] * θ[k]
    err_current = norm(fun_A(θ .+ 1e-5) - fun_A(θ) - dAh)
    # err_current = norm(fun_A(θ .+ 1e-5) - fun_A(θ))
    @info err_current
    err2 = max(err2, err_current)
end
@info "Multidimensional parameter test: error" err2

# err3 = 0.
# for i in 1:n, j in 1:n
#     global err3
#     deriv_Aij(θ) = deriv_A(θ)[i, j, :]
#     fun_Aij(θ) = fun_A(θ)[i, j, :]
#     θgrid = rand(Uniform(0.01,2), d, ntest)
#     err_ij = 0.
#     for k in 1:ntest
#         θ = θgrid[:, k]
#         dAh = dot(deriv_Aij(θ), θ)
#         err_current = norm(fun_Aij(θ .+ 1e-4) .- fun_Aij(θ) .- dAh)
#         err_ij = max(err_ij, err_current)
#     end
#     err3 = max(err3, err_ij)
# end
# @info "Multidimensional parameter test: error" err3