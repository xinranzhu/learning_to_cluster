using LinearAlgebra
using Plots
using MLDatasets
using Statistics
using Random
using Distances
using Distributions
using Roots
using ArgParse
using Clustering
using Combinatorics
using ProgressMeter
using Dates
using Printf
import Sobol: next!
using Sobol
using DataFrames
using CSV
using TensorOperations
using JLD
using Optim
using Arpack

include("../attributed/clusteringAttributed.jl")
include("../datastructs.jl")
include("model_reduction.jl")

# load Vhat 
Vhat_set = JLD.load("./saved_data/Vhat_set_false_1_2_1.jld")["data"]
Vhat = Vhat_set.Vhat
rangeθ = Vhat_set.rangeθ
k = Vhat_set.k
d = size(rangeθ, 1)
n = 35776
y = # partial true label
atttraindata = atttraindata(n, y, 0.1)

loss(θ::Vector) = loss_fun_reduction(n, k, θ, atttraindata, Vhat; if_deriv = false)[1] 
loss_deriv(θ::Vector) = loss_fun_reduction(n, k, θ, atttraindata, Vhat)[2] 

ntest = 10; h = 1e-5; hvec = h * ones(d)
# use quasi-random theta samples
s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
N = hcat([next!(s) for i = 1:ntest]...)' # ntest * d

err = 0.
before = Dates.now()
for i = 1:ntest
    θ = N[i, :]
    global err
    dlh = dot(loss_deriv(θ), hvec)
    err_current = norm(loss(θ .+ h) - loss(θ) - dlh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
    # @info err_current
    err = max(err, err_current)
end
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) 
@info "loss_deriv, multidimensional parameter: $err. Each evaluation took $(elapsedmin/ntest) min." 
