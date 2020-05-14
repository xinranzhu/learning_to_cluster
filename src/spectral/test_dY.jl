using LinearAlgebra
using MLDatasets
using Statistics
using Random
using Distances
using Distributions
using Roots
using Clustering
using Combinatorics
using ProgressMeter
using Distances
using Sobol: next!
using Sobol
using DataFrames
using CSV
using TensorOperations

include("comp_deriv.jl")
include("../kernels/kernels.jl")
include("spectral_reduction_main.jl")

# MNIST
# dataraw, _ = MNIST.testdata(Float64)
# data = reshape(permutedims(dataraw,(3, 1, 2)), size(dataraw, 3), size(dataraw,1)*size(dataraw,2));
# k = 10

# abalone
df = DataFrame(CSV.File("../datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8]) 
k = 29

# shuffle data: n * d
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :];
@info size(data)

X = data[1:3000, :]; n, d = size(X)
ntest = 10; h = 1e-5
I_rows = nothing

# compute Vhat and Vhatm
@info "Computing Vhat"
N_sample = 1000
ranges = [500 3000]
rangesm = repeat([500 3000], d, 1)

Vhat, _ = comp_Vhat(X, k, ranges; N_sample = 500)
@info "Size Vhat:" size(Vhat)
Vhatm, _ = comp_Vhat(X, k, rangesm; N_sample = 500)
@info "Size Vhatm:" size(Vhatm)

function comp_dY_full(X, θ, Vhat)
    dimθ = length(θ)
    L, dL = laplacian_L(X, θ)
    m = size(Vhat, 2)
    H = Vhat' * L * Vhat 
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
fun_Y(θ) = comp_dY_full(X, θ, Vhat)[1]
deriv_Y(θ) =  comp_dY_full(X, θ, Vhat)[2]
fun_Ym(θ) = comp_dY_full(X, θ, Vhatm)[1]
deriv_Ym(θ) =  comp_dY_full(X, θ, Vhatm)[2]

@info "Start 1d test"
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dYtest_fd = deriv_fd(fun_Y, h)
θgrid = range(500, stop=3000, length=ntest)
# θgrid = range(0.01, stop=2, length=ntest)
# err1 = maximum(norm.(deriv_L.(θgrid) - dLtest_fd.(θgrid))) # O(h^2)
err1 = norm(fun_Y.(θgrid .+ h) - fun_Y.(θgrid) - h .* deriv_Y.(θgrid)) # O(h^2)
# err1 = norm(fun_L.(θgrid .+ h) - fun_L.(θgrid .- h) - 2 * h .* deriv_L.(θgrid)) # O(h^3)
@info "dY, One-dim parameter: " err1

@info "Start multi-dim test"
m = size(Vhatm, 2)
θgrid = rand(Uniform(500, 3000), d, ntest)
# θgrid = rand(Uniform(0.01,2), d, ntest)
dYh = Array{Float64, 2}(undef, m, k)
err2 = 0.
hvec = h .* ones(d)
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    @tensor dYh[l, j] = deriv_Ym(θ)[l, j, k] * hvec[k]
    err_current = norm(fun_Ym(θ .+ h) - fun_Ym(θ) - dYh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
    @info err_current
    err2 = max(err2, err_current)
end
@info "dL, Multidimensional parameter: " err2


