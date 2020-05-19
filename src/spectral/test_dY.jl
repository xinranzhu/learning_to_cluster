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
using JLD

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
label = convert(Array, df[:, 9]) # 1 ~ 29
k = 29

# shuffle data: n * d
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :]
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

ntest = 10; h = 1e-5
I_rows = nothing

# load Vhat
Vhat_setm = JLD.load("../abalone/saved_data/Vhat_set_false_5_2.jld")["data"]
Vhatm = Vhat_setm.Vhat
Vhat_sets = JLD.load("../abalone/saved_data/Vhat_set_true_5_2.jld")["data"]
Vhats = Vhat_sets.Vhat

rangeθs = Vhat_sets.rangeθ
rangeθm = Vhat_setm.rangeθ
@assert d == size(rangeθm, 1)
@info "Finish loading Vhat, m_single = $(size(Vhats, 2)), m_multi =  $(size(Vhatm, 2))"


function comp_dY_full(X, θ, Vhat)
    dimθ = length(θ)
    L, dL = laplacian_L(X, θ)
    m = size(Vhat, 2)
    H = @views Vhat' * L * Vhat 
    ef = eigen(Symmetric(H), m-k+1:m)
    Y = ef.vectors
    Λ = ef.values
    if dimθ == 1
        dH = Vhat' * dL * Vhat 
    else
        dH = Array{Float64, 3}(undef, m, m, dimθ)
        @tensor dH[i,j,k] = Vhat'[i, s] * dL[s, l, k] * Vhat[l, j] 
    end
    dY = comp_dY(Y, Λ, H, dH, dimθ)
    return Y, dY
end


fun_Y(θ) = comp_dY_full(X, θ, Vhats)[1]
deriv_Y(θ) =  comp_dY_full(X, θ, Vhats)[2]
fun_Ym(θ) = comp_dY_full(X, θ, Vhatm)[1]
deriv_Ym(θ) =  comp_dY_full(X, θ, Vhatm)[2]

@info "Start 1d derivative test"
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dYtest_fd = deriv_fd(fun_Y, h)
θgrid = range(rangeθs[1], stop=rangeθs[2], length=ntest)
# θgrid = range(0.01, stop=2, length=ntest)
before = Dates.now()
# err1 = maximum(norm.(deriv_L.(θgrid) - dLtest_fd.(θgrid))) # O(h^2)
err1 = norm(fun_Y.(θgrid .+ h) - fun_Y.(θgrid) - h .* deriv_Y.(θgrid)) # O(h^2)
# err1 = norm(fun_L.(θgrid .+ h) - fun_L.(θgrid .- h) - 2 * h .* deriv_L.(θgrid)) # O(h^3)
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=3) 
@info "dY, one-dim parameter: $err1. Each evaluation took $(round(elapsedmin/ntest, digits=5)) min." 


@info "Start multi-dim derivative test"
m = size(Vhatm, 2)
θgrid = rand(Uniform(rangeθs[1], rangeθs[2]), d, ntest)
# θgrid = rand(Uniform(0.01,2), d, ntest)
dYh = Array{Float64, 2}(undef, m, k)
err2 = 0.
hvec = h .* ones(d)
before = Dates.now()
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    @tensor dYh[l, j] = deriv_Ym(θ)[l, j, k] * hvec[k]
    err_current = norm(fun_Ym(θ .+ h) - fun_Ym(θ) - dYh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
    @info err_current
    err2 = max(err2, err_current)
end
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=3) 
@info "dY, multidimensional parameter: $err2. Each evaluation took $(round(elapsedmin/ntest, digits=5)) min." 


# log
# julia test_dY.jl
# [ Info: (4177, 7)
# ┌ Info: Size of testing data
# └   (size(X), size(y)) = ((4177, 7), (4177,))
# ┌ Info: Size of training data
# └   ntrain = 417
# [ Info: Finish loading Vhat, m_single = 142, m_multi =  785
# [ Info: Start 1d derivative test
# [ Info: dY, one-dim parameter: 1.345496393027407e-7. Each evaluation took 0.962 min.
# [ Info: Start multi-dim derivative test
# [ Info: 4.614548970168324e-11
# [ Info: 1.8569207793155315e-11
# [ Info: 3.625247064623181e-11
# [ Info: 2.9040876370376112e-11
# [ Info: 3.1572482589790025e-11
# [ Info: 1.0579429267065698e-11
# [ Info: 5.6205517590192374e-11
# [ Info: 2.462204576917873e-11
# [ Info: 1.1767985271091683e-10
# [ Info: 1.9230913002572265e-11
# [ Info: dY, multidimensional parameter: 1.1767985271091683e-10. Each evaluation took 4.6432 min.