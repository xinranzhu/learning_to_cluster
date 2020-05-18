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

include("../TSNE/myTSNE.jl")
include("../kmeans_match_labels.jl")
include("../spectral/spectral_clustering.jl")
include("../spectral/spectral_reduction_main.jl")
include("../spectral/helpers.jl")
include("../datastructs.jl")

s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--ntotal"
        help = "size of testing set"
        arg_type = Int
        default = 4177
    "--TSNE"
        help = "use TSNE or not"
        action = :store_true
    "--dimY" 
        help = "target dimension in TSNE"
        arg_type = Int
        default = 2
    "--spectral"
        help = "use TSNE or not"
        action = :store_true
    "--specparam"
        help = "silimarity parameter in spectral clustering, train if 0"
        arg_type = Float64
        default = 0.1
    "--trainratio"
        help = "ratio of training set to testing set"
        arg_type = Float64
        default = 0.1
    "--reduction"
        help = "use model reduction or not"
        action = :store_true
    "--dataset"
        help = "which dataset to use"
        arg_type = String
        default = "abalone"
    "--single"
        help = "use single θ or multi-dim"
        action = :store_true
    "--set_Nsample"
        help = "use which settings of N_sample -- see precompute_Vhat"
        arg_type = Int
        default = 0
    "--set_range"
        help = "use which settings of rangeθ -- see precompute_Vhat"
        arg_type = Int
        default = 0
    "--rangetheta"
        help = "mannually set rangeθ"
        arg_type = Float64
        nargs = 2
        default = [0.1, 2000.] # for abalone 
end
parsed_args = parse_args(ARGS, s)

# input data
# if parsed_args["dataset"] == "abalone"
# abalone
df = DataFrame(CSV.File("../datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8]) 
label = convert(Array, df[:, 9]) # 1 ~ 29
k = 29
# elseif parsed_args["dataset"] == "MNIST"
#     dataraw, label = MNIST.testdata(Float64)
#     data = reshape(permutedims(dataraw,(3, 1, 2)), size(dataraw, 3), size(dataraw,1)*size(dataraw,2));
#     k = 10
#     label .+= 1 # 1 ~ 10
# else
#     throw(ArgumentError("Dataset not supported."))
# end
# shuffle data: n * d
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :]
label = label[ind_shuffle]
@info "Size of whole dataset: " size(data), size(label)

# select from data and label
n = parsed_args["ntotal"]
if n > size(data, 1)
    @warn "Have $(size(data, 1)) data only"
    n = size(data, 1)
end
testdata = testingData(data[1:n, :], label[1:n])
X = testdata.X; y = testdata.y
d = testdata.d
@info "Size of testing data" size(X), size(y)
traindata = trainingData(X, y, parsed_args["trainratio"])
ntrain = traindata.n
@info "Size of training data" size(traindata.X), size(traindata.y), typeof(traindata)

range_set = JLD.load("./saved_data/abalone_range_set.jld")["data"]
if parsed_args["set_range"] == 0 # if no specific range setting is selected
    rangeθs = reshape(convert(Array{Float64, 1}, parsed_args["rangetheta"]), 1, 2)
else # use the assigned setting
    rangeθs = reshape(range_set[parsed_args["set_range"], :], 1, 2)
end
rangeθm = repeat(rangeθs, d, 1)
rangeθ = parsed_args["single"] ? rangeθs : rangeθm
dimθ = size(rangeθ, 1)

before = Dates.now()
Vhat_timecost = 0.; N_sample = 0.; m = 0.; 
if parsed_args["reduction"]
    global Vhat_timecost, N_sample, m
    println("Start spectral clustering with model reduction")
    if ntrain == 0
        R = spectral_reduction_main(X, k, parsed_args["specparam"], rangeθ)
        algorithm = "Spectral clustering with model reduction (fixed θ = $(parsed_args["specparam"]))"
    else 
        if parsed_args["set_range"] == 0 || parsed_args["set_Nsample"] == 0
            Vhat_set = nothing
        else  # load Vhat_set if precomputed
            @info "Loading Vhat"
            Vhat_set = JLD.load("./saved_data/Vhat_set_$(parsed_args["single"])_$(parsed_args["set_range"])_$(parsed_args["set_Nsample"]).jld")["data"]
            N_sample = Vhat_set.N_sample
            Vhat_timecost = Vhat_set.timecost
            m = size(Vhat_set.Vhat, 2)
            @assert Vhat_set.rangeθ == rangeθ 
            @info "Finish loading Vhat, rangeθ, N_sample, m, timecost" rangeθ[1, 1], rangeθ[1, 2], N_sample, Vhat_timecost
        end
        θ_init = reshape(rand(Uniform(rangeθ[1, 1], rangeθ[1, 2]), dimθ, 1), dimθ)
        θ_init = dimθ == 1 ? θ_init[1] : θ_init 
        assignment, θ = spectral_reduction_main(X, k, θ_init, rangeθ; traindata = traindata, Vhat_set = Vhat_set)
        algorithm = "Spectral clustering with model reduction (trained θ = $(θ))"
    end
else # do normal clustering, using plain kmeans or TSNE+kmeans or spectral clustering + kmeans
    algorithm = "Kmeans"
    if parsed_args["TSNE"]
        println("Start TSNE")
        X = mytsne(X, parsed_args["dimY"], calculate_error_every=100, plot_every=0, lr=100)
        algorithm = "TSNE(dim=$(parsed_args["dimY"])) + Kmeans"
    elseif parsed_args["spectral"] 
        println("Start spectral clustering")
        if parsed_args["specparam"] > 0 # if given a fixed param
            X, _ = spectral_clustering_model(X, k, parsed_args["specparam"]; if_deriv = false) 
            algorithm = "Spectral (fixed θ=$(parsed_args["specparam"])) + Kmeans"
        else
            X, θ = spectral_clustering_main(X, k, traindata, rangeθ) 
            algorithm = "Supervised Spectral (trained θ=$θ) + Kmeans"
        end
    end
    # K-means clustering
    println("Start K-means")
    # warning the matrix put into kmeans should be d*N
    R = kmeans(X', k; maxiter=200, display=:final)
    @assert nclusters(R) == k # verify the number of clusters
    assignment = assignments(R) # get the assignments of points to clusters
end
# c = counts(R) # get the cluster sizes
# M = R.centers # get the cluster centers
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) + Vhat_timecost 

# assign labels and return accuracy
# max_acc, matched_assignment = ACC_match_labels(assignment, y, k, parsed_args["dataset"])
@info "Matching labels, ntrain:" ntrain
max_acc, matched_assignment = bipartite_match_labels(assignment, y, k; ntrain = ntrain) # assignment is updated
RI = randindex(matched_assignment[ntrain+1:end], y[ntrain+1:end])


io = open("$(parsed_args["dataset"])_results.txt", "a")
write(io, "\n$(Dates.now()), randseed: $randseed \n" )
write(io, "Data set: $(parsed_args["dataset"])  testing points: $n; training data: $ntrain\n") 
write(io, "rangeθ: $rangeθs, dimθ: $dimθ; N_sample: $N_sample; m: $m ; Vhat_timecost = $Vhat_timecost \n")
write(io, "Algorithm: $algorithm 
    Time cost:                                   $(@sprintf("%.5f", elapsedmin))
    Accuracy (ACC):                              $(@sprintf("%.5f", max_acc))
    Adjusted Rand index:                         $(@sprintf("%.5f", RI[1]))
    Rand index (agreement probability):          $(@sprintf("%.5f", RI[2]))
    Mirkin's index (disagreement probability):   $(@sprintf("%.5f", RI[3]))
    Hubert's index P(agree) - P(disagree):       $(@sprintf("%.5f", RI[4])) \n")
close(io)