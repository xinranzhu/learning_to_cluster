using LinearAlgebra
using MLDatasets
using Statistics
using Random
using Distances
using Distributions
using Roots
using Clustering
using Combinatorics
using Distances
using Sobol: next!
using Sobol
using DataFrames
using CSV
using TensorOperations

# include("comp_deriv.jl")
# include("../kernels/kernels.jl")
include("spectral_reduction_main.jl")

# MNIST
# dataraw, _ = MNIST.testdata(Float64)
# data = reshape(permutedims(dataraw,(3, 1, 2)), size(dataraw, 3), size(dataraw,1)*size(dataraw,2));
# k = 10

# abalone
df = DataFrame(CSV.File("../datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8]) 
label = convert(Array, df[:, 2]) 
k = 29

@info "Size of whole dataset: " size(data), size(label)

# shuffle data: n * d
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :];
label = label[ind_shuffle]
@info size(data)

X = data[1:4177, :]; n, d = size(X)
y = label[1:4177]
@info "Size of testing data" size(X), size(y)
# select a fraction to be training data
ntrain = Int(floor(n*0.2))
idtrain = 1:ntrain
Xtrain = X[idtrain, :]
ytrain = y[idtrain]

# compute Vhat and Vhatm
@info "Computing Vhat"
N_sample = 500
rangeθ = [1 3000]
ranges = rangeθ
rangesm = repeat(rangeθ, d, 1)

Vhat, _ = comp_Vhat(X, k, ranges; N_sample = 500)
@info "Size Vhat:" size(Vhat)
Vhatm, _ = comp_Vhat(X, k, rangesm; N_sample = 500)
@info "Size Vhatm:" size(Vhatm)

Apm = gen_constraints(Xtrain, ytrain) 

loss(θ) = loss_fun_reduction(θ, X, Xtrain, idtrain, Apm, k, Vhat; if_deriv = false)[1] 
loss_deriv(θ) = loss_fun_reduction(θ, X, Xtrain, idtrain, Apm, k, Vhat; if_deriv = true)[2] 

lossm(θ) = loss_fun_reduction(θ, X, Xtrain, idtrain, Apm, k, Vhatm; if_deriv = false)[1] 
loss_derivm(θ) = loss_fun_reduction(θ, X, Xtrain, idtrain, Apm, k, Vhatm; if_deriv = false)[2] 

@info "Start 1d derivative test"
ntest = 10; h = 1e-5
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dYtest_fd = deriv_fd(loss, h)
θgrid = range(ranges[1], stop=ranges[2], length=ntest)
# err1 = maximum(norm.(deriv_L.(θgrid) - dLtest_fd.(θgrid))) # O(h^2)
err1 = norm(loss.(θgrid .+ h) - loss.(θgrid) - h .* loss_deriv.(θgrid)) # O(h^2)
# err1 = norm(fun_L.(θgrid .+ h) - fun_L.(θgrid .- h) - 2 * h .* deriv_L.(θgrid)) # O(h^3)
@info "loss_deriv, one-dim parameter: " err1

@info "Start multi-dim derivative test"
m = size(Vhatm, 2)
θgrid = rand(Uniform(ranges[1], ranges[2]), d, ntest)
err2 = 0.
hvec = h .* ones(d)
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    dlh = dot(loss_derivm(θ), hvec)
    err_current = norm(lossm(θ .+ h) - lossm(θ) - dlh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
    # @info err_current
    err2 = max(err2, err_current)
end
@info "loss_deriv, multidimensional parameter: " err2

