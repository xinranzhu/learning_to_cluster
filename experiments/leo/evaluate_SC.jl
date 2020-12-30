using Distributed
addprocs(8)
@everywhere using Distributed
@everywhere using SharedArrays
@everywhere using TimerOutputs
@everywhere const to = TimerOutput()
@everywhere using CSV
@everywhere using DataFrames
@everywhere using Clustering
@everywhere using Hungarian
@everywhere using Combinatorics
@everywhere using Statistics
@everywhere using LinearAlgebra
@everywhere using LineSearches
@everywhere using Optim
@everywhere using Plots

@everywhere include("src/l2c.jl")

# load data
@everywhere mydataset = load("experiments/datasets/derived_datasets/abalone_2_class.jld")
@everywhere label = mydataset["label"]
@everywhere data = mydataset["data"]
@everywhere k = mydataset["k"]

# run spectral clustering evaluator
@everywhere function populate(S)
        for i in localindices(S)
            #S[i] = myid()
            #S[i] = norm(2*test.a[:, i])
            res = evaluate_spectral_clustering(data, label; frac_train = 2*i * 0.1 - 0.1, #1, 3, 5, 7, 9
                train = false, normalized = true, ntrials = 20, time_limit = 200)
            S[i] = round(res[:Ans], digits = 4)
            #S[i+1] = res[:N]
        end
    end

print("procs: "* string(procs()))
@timeit to "parallel" S = SharedArray{Float64,2}((1,5), init = S -> populate(S))

# write results
io = open("12_29_20_results_normalized_untrained_better_separated_abalone_2_class.txt", "a")
write(io, "\n SHARED ARRAY: UNNORMALIZED SC ACCS\n" )
for i = 1:length(S)
    write(io, "\n $(S[i]) " )
end
close(io)

R = kmeans(data', k; maxiter=100, display=:final)
@assert nclusters(R) == k # verify the number of clusters
assignment = assignments(R) # get the assignments of points to clusters
max_acc, matched_assignment = bipartite_match_labels(assignment, label, k)
@show max_acc
