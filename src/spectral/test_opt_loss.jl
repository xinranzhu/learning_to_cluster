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
using JuMP, Ipopt
using Arpack
using Optim
# import NLopt.optimize!

include("../TSNE/myTSNE.jl")
include("../kmeans_match_labels.jl")
include("./spectral_clustering.jl")
include("./spectral_reduction_main.jl")
include("../datastructs.jl")


s = ArgParseSettings()
@add_arg_table! s begin
    "--set_Nsample"
        help = "use which settings of N_sample -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--set_range"
        help = "use which settings of rangeθ -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--relabel"
        help = "relabel or not"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

df = DataFrame(CSV.File("../datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8]) 
label = convert(Array, df[:, 9]) # 1 ~ 29
k = 29

# relabel: view <= 5 as one cluster, and >=15 as one cluster
if parsed_args["relabel"]
    label[label .<= 5] .= 5
    label[label .>= 15] .= 15
    label .-= 4
    k = 11
end
@info "Target number of clusterings $k"

randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :]
label = label[ind_shuffle]
@info "Size of whole dataset: " size(data), size(label)

n = 4177

testdata = testingData(data[1:n, :], label[1:n])
X = testdata.X; y = testdata.y
d = testdata.d
@info "Size of testing data" size(X), size(y)
traindata = trainingData(X, y, 200)
ntrain = traindata.n
Apm = traindata.Apm
@info "Size of training data" size(traindata.X), size(traindata.y), typeof(traindata)

if parsed_args["relabel"]
    Vhat_set = JLD.load("../abalone/saved_data/Vhat_set_$(parsed_args["relabel"])_$(parsed_args["set_range"])_$(parsed_args["set_Nsample"]).jld")["data"]
else

Vhat = Vhat_set.Vhat
timecost = Vhat_set.timecost
I_rows = Vhat_set.I_rows
m = size(Vhat, 2)
@info "Finish loading Vhat, size of Vhat and time cost" size(Vhat), timecost

rangeθ = Vhat_set.rangeθ
dimθ = size(rangeθ, 1)
@info "n=$n, dimθ=$dimθ"
@info "rangeθ=$rangeθ"

θ_init = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]

loss(θ) = loss_fun_reduction(X, k, θ, traindata, Vhat; if_deriv = false)[1] 
loss_deriv(θ) = loss_fun_reduction(X, k, θ, traindata, Vhat)[2] 
function loss_deriv!(g, θ)
    g = loss_deriv(θ)
end

inner_optimizer = LBFGS()
@info "Start optimization"
@time results = Optim.optimize(loss, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer))
θ = Optim.minimizer(results)

@info "final θ" θ