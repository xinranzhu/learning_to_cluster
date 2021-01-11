using Statistics
using Random
using Distances
using Distributions
using Roots
using ArgParse
using Combinatorics
using Dates
using Printf
import Sobol: next!
using Sobol
using DataFrames
using CSV
using TensorOperations
using JLD
using LinearAlgebra

include("../attributed/attributed.jl")
include("../datastructs.jl")
include("./model_reduction.jl")

const dimθ = 12
const n = 35776
# load pre-set ranges and N_samples
ranges = load("./saved_data/reddit_range_set.jld")["data"]
N_sample_set = load("./saved_data/reddit_Nsample_set.jld")["data"]
k_set = load("./saved_data/reddit_k_set.jld")["data"]
@info ranges
@info N_sample_set

n_range = size(ranges, 1)
n_N_sample = length(N_sample_set)
n_k = length(k_set)
@info n_range, n_N_sample, n_k
for j = 1
    N_sample = N_sample_set[j]
    for s = 3
        k = k_set[s]
        for i = 1
            rangeθ = ranges[i, :, :]
            @assert size(rangeθ) == (dimθ, 2)
            @info "Start computing Vhat, range $i and N_sample = $N_sample, k = $k"
            before = Dates.now()
            Vhat, s = comp_Vhat(n, k, rangeθ; N_sample = N_sample) 
            after = Dates.now()
            elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
            m = size(Vhat, 2)
            @info "k = $k, N = $N_sample, m = $m and s = $s"
            @assert m > k
            Vhat_set = (Vhat = Vhat, rangeθ = rangeθ, N_sample = N_sample, timecost = elapsedmin, k = k)
            save("./saved_data/Vhat_set_test_$(i)_$(j)_$(s).jld", "data", Vhat_set)
            @info "Finish computing Vhat size, time cost", size(Vhat), elapsedmin
        end
    end
end
