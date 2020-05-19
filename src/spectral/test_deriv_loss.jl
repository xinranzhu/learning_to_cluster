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
using Arpack
using JLD

include("comp_deriv.jl")
include("helpers.jl")
include("../datastructs.jl")
include("../kernels/kernels.jl")
include("spectral_reduction_main.jl")

# MNIST
# dataraw, _ = MNIST.testdata(Float64)
# data = reshape(permutedims(dataraw,(3, 1, 2)), size(dataraw, 3), size(dataraw,1)*size(dataraw,2));
# k = 10

# abalone
df = DataFrame(CSV.File("../datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8]) 
label = convert(Array, df[:, 9]) 
k = 29

@info "Size of whole dataset: " size(data), size(label)

# shuffle data: n * d
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :];
label = label[ind_shuffle]
@info size(data)

n = 4177
testdata = testingData(data[1:n, :], label[1:n])
X = testdata.X; y = testdata.y
d = testdata.d; n = testdata.n 
@info "Size of testing data" size(X), size(y)
traindata = trainingData(X, y, 0.1)
ntrain = traindata.n
Apm = traindata.Apm
@info "Size of training data" ntrain

# load Vhat
Vhat_setm = JLD.load("../abalone/saved_data/Vhat_set_false_5_2.jld")["data"]
Vhatm = Vhat_setm.Vhat
Vhat_sets = JLD.load("../abalone/saved_data/Vhat_set_true_5_2.jld")["data"]
Vhats = Vhat_sets.Vhat

rangeθs = Vhat_sets.rangeθ
rangeθm = Vhat_setm.rangeθ
@assert d == size(rangeθm, 1)
@info "Finish loading Vhat, m_single = $(size(Vhats, 2)), m_multi =  $(size(Vhatm, 2))"

loss(θ::Float64) = loss_fun_reduction(X, k, θ, traindata, Vhats; if_deriv = false)[1] 
loss_deriv(θ::Float64) = loss_fun_reduction(X, k, θ, traindata, Vhats; if_deriv = true)[2] 

lossm(θ::Array{Float64,1}) = loss_fun_reduction(X, k, θ, traindata, Vhatm; if_deriv = false)[1] 
loss_derivm(θ::Array{Float64,1}) = loss_fun_reduction(X, k, θ, traindata, Vhatm; if_deriv = true)[2] 

@info "Start 1d derivative test"
ntest = 10; h = 1e-5
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dYtest_fd = deriv_fd(loss, h)
θgrid = range(rangeθs[1], stop=rangeθs[2], length=ntest)
before = Dates.now()
# err1 = maximum(norm.(deriv_L.(θgrid) - dLtest_fd.(θgrid))) # O(h^2)
err1 = norm(loss.(θgrid .+ h) - loss.(θgrid) - h .* loss_deriv.(θgrid)) # O(h^2)
err2 = norm(loss.(θgrid .+ h) - loss.(θgrid .- h) - 2h .* loss_deriv.(θgrid)) # O(h^3)
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=3) 
@info "loss_deriv, one-dim parameter: $err1, $err2. Each evaluation took $(elapsedmin/ntest) min." 

@info "Start multi-dim derivative test"
m = size(Vhatm, 2)
θgrid = rand(Uniform(rangeθs[1], rangeθs[2]), d, ntest)
err2 = 0.
hvec = h .* ones(d)
before = Dates.now()
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    dlh = dot(loss_derivm(θ), hvec)
    err_current = norm(lossm(θ .+ h) - lossm(θ) - dlh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
    # @info err_current
    err2 = max(err2, err_current)
end
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=3) 
@info "loss_deriv, multidimensional parameter: $err2. Each evaluation took $(elapsedmin/ntest) min." 

