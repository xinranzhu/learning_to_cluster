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

include("../attributed/attributed.jl")
include("../datastructs.jl")
include("model_reduction.jl")

s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--set_range"
        help = "use which settings of rangeθ -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--set_Nsample"
        help = "use which settings of N_sample -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--set_k"
        help = "use which settings of k -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--idtheta"
        help = "assigned value of theta, skip training -- should be load from file"
        arg_type = Int
        default = 0
end
parsed_args = parse_args(ARGS, s)

# load Vhat 
Vhat_set = JLD.load("./saved_data/Vhat_set_1_1_2.jld")["data"]
Vhat = Vhat_set.Vhat
rangeθ = Vhat_set.rangeθ
k = Vhat_set.k
d = size(rangeθ, 1)
@info "Finish loading Vhat, k=$k, dtheta = $d"
n = 35776
(idtrain, ytrain) = trainInfo_fixed()
ntrain = length(idtrain)
traindata = atttraindata(ntrain, idtrain, ytrain)
@info "Finish build training data, ntrain=$ntrain, type of traindata = $(typeof(traindata))."

loss(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat; if_deriv = false)[1] 
loss_deriv(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat)[2] 

ntest = 10; h = 1e-5; hvec = h * ones(d)
# use quasi-random theta samples
s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
N = hcat([next!(s) for i = 1:ntest]...)' # ntest * d
@info "Size of testing theta grid: $(size(N))"

err = 0.
before = Dates.now()
for i = 1:ntest
    θ = N[i, :]
    global err
    dlh = dot(loss_deriv(θ), hvec)
    err_current = norm(loss(θ .+ h) - loss(θ) - dlh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
    @info "$i test: current error = $err_current"
    err = max(err, err_current)
end
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) 
@info "loss_deriv, multidimensional parameter: $err. Each evaluation took $(elapsedmin/ntest) min." 
