# precompute Vhat for a bunch of candidate ranges
# abalone 

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

include("../TSNE/myTSNE.jl")
include("../kmeans_match_labels.jl")
include("../spectral/spectral_clustering.jl")
include("../spectral/spectral_reduction_main.jl")

s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--single"
        help = "use single θ or multi-dim"
        action = :store_true
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

n = 4177
@assert n <= size(data, 1) 
X = data[1:n, :]
y = label[1:n]
d = size(X, 2)
@info "Loading data: \nSize of whole dataset: " size(data), size(label)
@info "Size of testing data" size(X), size(y)

# load range set and Nsample set
ranges = load("./saved_data/abalone_range_set.jld")["data"]
N_sample_set = load("./saved_data/abalone_Nsample_set.jld")["data"]
# ranges = [0.1 200.;
#         0.1 1000; 
#         0.1 1500; 
#         1 200;
#         1 1000
#         1 1500
#         0.1, 30]
# N_sample_set = [100, 500, 1000, 1500]

n_range = size(ranges, 1)
n_N_sample = length(N_sample_set)
@info "size(range_set) = $(size(ranges)); size(N_sample_set) = $(size(N_sample_set))"

for j in 1:3
    for i in [4, 5, 7]
        rangeθs = reshape(ranges[i, :], 1, 2)
        rangeθ = parsed_args["single"] ? rangeθs : repeat(rangeθs, d, 1)
        N_sample = N_sample_set[j]
        @info "Start computing Vhat, range, single or multi and N_sample", rangeθs, parsed_args["single"], N_sample
        before = Dates.now()
        Vhat, I_rows = comp_Vhat(X, k, rangeθ; N_sample = N_sample) 
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        m = size(Vhat, 2)
        @assert m > k 
        Vhat_set = (Vhat = Vhat, rangeθ = rangeθ, I_rows = I_rows, N_sample = N_sample, timecost = elapsedmin)
        save("./saved_data/Vhat_set_$(parsed_args["relabel"])_$(parsed_args["single"])_$(i)_$(j).jld", "data", Vhat_set)
        @info "Finish computing Vhat size, time cost", size(Vhat), elapsedmin
    end
end
