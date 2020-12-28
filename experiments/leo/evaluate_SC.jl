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
@everywhere begin
    df = DataFrame(CSV.File("experiments/datasets/abalone.csv", header = 0))
    data = convert(Matrix, df[500:2500,2:8])
    label = convert(Array, df[500:2500, 9]) # 1 ~ 29
    # relabel: regroup labels <= 5 as one lable, and >=15 as one label
    # then target number of clusters = 11
    label[label .<= 6] .= 1
    label[(&).(label .> 6, label .<=8)] .= 2
    label[(&).(label .> 8, label .<=10)] .= 3
    label[(&).(label .> 10, label .<=12)] .= 4
    label[(&).(label .> 12, label .<=14)] .= 5
    label[label .> 14] .= 6
    k = 6
end

# run spectral clustering evaluator
@everywhere function populate(S)
        for i in localindices(S)
            #S[i] = myid()
            #S[i] = norm(2*test.a[:, i])
            res = evaluate_spectral_clustering(data, label; frac_train = i * 0.1,
                train = true, normalized = true, ntrials = 20, time_limit = 100)
            S[i] = res[:Ans]
            #S[i+1] = res[:N]
        end
    end

print("procs: "* string(procs()))
@timeit to "parallel" S = SharedArray{Float64,2}((1,9), init = S -> populate(S))

# write results
io = open("12_28_20_results_normalized_new_test.txt", "a")
write(io, "\n SHARED ARRAY: UNNORMALIZED SC ACCS\n" )
for i = 1:length(S)
    write(io, "\n $(S[i]) " )
end
close(io)
