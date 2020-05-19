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

# load Vhat
Vhat_setm = JLD.load("../abalone/saved_data/Vhat_set_false_5_2.jld")["data"]
Vhatm = Vhat_setm.Vhat
Vhat_sets = JLD.load("../abalone/saved_data/Vhat_set_true_5_2.jld")["data"]
Vhats = Vhat_sets.Vhat

rangeθs = Vhat_sets.rangeθ
rangeθm = Vhat_setm.rangeθ
@assert d == size(rangeθm, 1)
@info "Finish loading Vhat, m_single = $(size(Vhats, 2)), m_multi =  $(size(Vhatm, 2))"

ntest = 10; h = 1e-5; hvec = h .* ones(d)
I_rows = nothing

fun_A(θ) = affinity_A(X, θ; if_deriv = false)[1]
deriv_A(θ) =   affinity_A(X, θ)[2]
fun_D(θ) = degree_D(X, θ; if_deriv = false)[1]
deriv_D(θ) =  degree_D(X, θ)[2]
fun_L(θ) = laplacian_L(X, θ; I_rows = I_rows, if_deriv = false)[1]
deriv_L(θ) =  laplacian_L(X, θ; I_rows = I_rows)[2]

θgrids = range(rangeθs[1], stop=rangeθs[2], length=ntest)
θgridm = rand(Uniform(rangeθs[1],rangeθs[2]), d, ntest)


# 1d parameter
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dAtest_fd = deriv_fd(fun_A, h)
dDtest_fd = deriv_fd(fun_D, h)
dLtest_fd = deriv_fd(fun_L, h)

@time errA = maximum(norm.(deriv_A.(θgrids) - dAtest_fd.(θgrids))) # O(h^2)
@time errD = maximum(norm.(deriv_D.(θgrids) - dDtest_fd.(θgrids))) # O(h^2)
@time errL = maximum(norm.(deriv_L.(θgrids) - dLtest_fd.(θgrids))) # O(h^2)
@info "One-dim parameter: errA = $errA, errD = $errD, errL = $errL" errA, errD, errL


# multi-dimensional parameter
n1 = I_rows == nothing ? n : length(I_rows)
dAh = Array{Float64, 2}(undef, n1, n)
dDh = similar(dAh)
dLh = similar(dAh)
errA = 0.
errD = 0.
errL = 0.
for i in 1:ntest
    global errA, errD, errL
    θ = θgridm[:, i]
    @tensor dAh[l, j] = deriv_A(θ)[l, j, k] * hvec[k]
    dDh = deriv_D(θ) * hvec
    @tensor dLh[l, j] = deriv_L(θ)[l, j, k] * hvec[k]
    err_curA = norm(fun_A(θ .+ h) - fun_A(θ) - dAh) # O(h^2)
    err_curD = norm(fun_D(θ .+ h) - fun_D(θ) - dDh)  # O(h^2)
    err_curL = norm(fun_L(θ .+ h) - fun_L(θ) - dLh) # O(h^2)
    @info "Current error: err_A = $err_curA, err_D = $err_curD, err_L = $err_curL "
    errA = max(errA, err_curA)
    errD = max(errD, err_curD)
    errL = max(errL, err_curL)
end
@info "Multi-dim parameter: errA = $errA, errD = $errD, errL = $errL" errA, errD, errL

# log 
# julia test_deriv_matrices.jl
# [ Info: (4177, 7)
# ┌ Info: Size of testing data
# └   (size(X), size(y)) = ((4177, 7), (4177,))
# ┌ Info: Size of training data
# └   ntrain = 417
# [ Info: Finish loading Vhat, m_single = 142, m_multi =  785
#  48.579015 seconds (11.13 M allocations: 53.832 GiB, 6.86% gc time)
#  40.483353 seconds (1.86 M allocations: 48.192 GiB, 5.91% gc time)
#  64.729067 seconds (4.26 M allocations: 66.507 GiB, 6.03% gc time)
# ┌ Info: One-dim parameter: errA = 2.0068380368752932e-8, errD = 2.580351292053389e-5, errL = 3.297137212951083e-10
# └   (errA, errD, errL) = (2.0068380368752932e-8, 2.580351292053389e-5, 3.297137212951083e-10)
# [ Info: Current error: err_A = 3.373001220401634e-11, err_D = 1.857413123547252e-10, err_L = 1.9354770429448522e-12
# [ Info: Current error: err_A = 4.5553577286023705e-11, err_D = 2.2688004046565033e-10, err_L = 2.6688637438024964e-12
# [ Info: Current error: err_A = 3.565354381064998e-11, err_D = 2.003851198214921e-10, err_L = 1.866670074819277e-12
# [ Info: Current error: err_A = 2.783385384975883e-11, err_D = 1.5793233360123588e-10, err_L = 1.40845617728551e-12
# [ Info: Current error: err_A = 2.3952137816921403e-11, err_D = 1.5934922688667695e-10, err_L = 1.006656935720164e-12
# [ Info: Current error: err_A = 4.185000623004004e-11, err_D = 1.8332344261032658e-10, err_L = 3.865099192499428e-12
# [ Info: Current error: err_A = 4.632363668667476e-11, err_D = 2.2024616295706713e-10, err_L = 3.2424256470666005e-12
# [ Info: Current error: err_A = 4.773026516877381e-11, err_D = 2.400309966025602e-10, err_L = 3.1084501228147102e-12
# [ Info: Current error: err_A = 4.1041700565689654e-11, err_D = 2.0195594969630306e-10, err_L = 3.2038761799110094e-12
# [ Info: Current error: err_A = 4.014622912282196e-11, err_D = 2.0257488122178348e-10, err_L = 2.707833399093937e-12
# ┌ Info: Multi-dim parameter: errA = 4.773026516877381e-11, errD = 2.400309966025602e-10, errL = 3.865099192499428e-12
# └   (errA, errD, errL) = (4.773026516877381e-11, 2.400309966025602e-10, 3.865099192499428e-12)