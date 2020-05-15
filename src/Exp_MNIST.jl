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

include("TSNE/myTSNE.jl")
include("kmeans_match_labels.jl")
include("spectral/spectral_clustering.jl")
include("spectral/helpers.jl")

s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--ntotal"
        help = "size of testing set"
        arg_type = Int
        default = 2500
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
        default = 0.05
    "--reduction"
        help = "use model reduction or not"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)
# input data
dataraw, label = MNIST.traindata(Float64)
ntotal = parsed_args["ntotal"]
randseed = 1234; rng = MersenneTwister(randseed)
idx = randperm(rng, ntotal)
data = reshape(permutedims(dataraw[:, :, idx],(3, 1, 2)), ntotal, size(dataraw,1)*size(dataraw,2))
label = label[idx]
# select a fraction to be training data
ntrain = Int(floor(ntotal*parsed_args["trainratio"]))
Xtrain = data[1:ntrain, :]
labeltrain = label[1:ntrain]
idtrain = 1:ntrain
k = 10 # number of target clustering

before = Dates.now()
if parsed_args["reduction"]
    println("Start spectral clustering with model reduction")
    if ntrain == 0
        R = spectral_reduction_main(data, k, parsed_args["specparam"])
        algorithm = "Spectral clustering with model reduction (fixed θ = $(parsed_args["specparam"]))"
    else 
        θ_init = parsed_args["specparam"] # or random
        R, θ = spectral_reduction_main(data, k, parsed_args["specparam"], Xtrain = Xtrain, ytrain = labeltrain)
        algorithm = "Spectral clustering with model reduction (trained θ = $(θ))"
    end
else # do normal clustering, using plain kmeans or TSNE+kmeans or spectral clustering + kmeans
    X = data # set of nodes put into kmeans, unprocessed
    algorithm = "Kmeans"
    if parsed_args["TSNE"]
        println("Start TSNE")
        X = mytsne(data, parsed_args["dimY"], calculate_error_every=100, plot_every=0, lr=100)
        algorithm = "TSNE(dim=$(parsed_args["dimY"])) + Kmeans"
    elseif parsed_args["spectral"] 
        println("Start spectral clustering")
        if ntrain == 0 # if given a fixed param
            X = spectral_clustering_model(data, k, parsed_args["specparam"]) 
            algorithm = "Spectral (fixed θ=$(parsed_args["specparam"])) + Kmeans"
        else
            X, θ = spectral_clustering_main(data, Xtrain, labeltrain, k) #TODO 
            algorithm = "Supervised Spectral (trained θ=$θ) + Kmeans"
        end
    end
    # K-means clustering
    println("Start K-means")
    # warning the matrix put into kmeans should be d*N
    R = kmeans(X', k; maxiter=200, display=:final)
end

@assert nclusters(R) == k # verify the number of clusters
assignment = assignments(R) # get the assignments of points to clusters
# c = counts(R) # get the cluster sizes
# M = R.centers # get the cluster centers
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

# assign labels and return accuracy
max_acc, matched_assignment, matched_actual_labels = ACC_match_labels(assignment, label, k)
RI = randindex(matched_assignment, matched_actual_labels)

io = open("MNIST_results.txt", "a")
write(io, "\n$(Dates.now()), randseed: $randseed \n" )
write(io, "Data set: MNIST  number of testing points: $ntotal \n") 
write(io, "Algorithm: $algorithm 
    Time cost:                                   $(@sprintf("%.5f", elapsedmin))
    Accuracy (ACC):                              $(@sprintf("%.5f", max_acc))
    Adjusted Rand index:                         $(@sprintf("%.5f", RI[1]))
    Rand index (agreement probability):          $(@sprintf("%.5f", RI[2]))
    Mirkin's index (disagreement probability):   $(@sprintf("%.5f", RI[3]))
    Hubert's index P(agree) - P(disagree):       $(@sprintf("%.5f", RI[4])) \n")
close(io)