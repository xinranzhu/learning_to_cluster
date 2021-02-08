
using LinearAlgebra
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
using KrylovKit


include("model_reduction.jl")
include("../datastructs.jl")
include("../attributed/attributed.jl")
include("../conductance/conductance.jl")

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

const n = 35776
const dimθ = 12
const ntrain = 1662

rid = parsed_args["set_range"]
Nid = parsed_args["set_Nsample"]
kid = parsed_args["set_k"]
θ = nothing; traindata = nothing

# load Vhat
Vhat_set = JLD.load("./saved_data/Vhat_set_$(rid)_$(Nid)_$(kid).jld")["data"]
k = Vhat_set.k
rangeθ = Vhat_set.rangeθ
dimθ = size(rangeθ, 1)
N_sample = Vhat_set.N_sample
Vhat_timecost = Vhat_set.timecost
m = size(Vhat_set.Vhat, 2)
# @assert sum(θ .> rangeθ[:, 1]) + sum(θ .< rangeθ[:, 2]) == 2*dimθ
@info "Finish loading Vhat, N_sample, m, k,timecost" N_sample, m, k, Vhat_timecost

# clustering randomly
before = Dates.now()
assignment = rand([1:k...], n)
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) + Vhat_timecost 

# spectral clustering with theta = 0
@info "Start spectral clustering with theta = 0"
before = Dates.now()
L, _ = laplacian_attributed_L(zeros(dimθ); if_deriv = false)
_, Vk = eigsolve(L, k, :LR, Float64; issymmetric=true, tol = 1e-16)
V = hcat(Vk...)
@info size(V), n, k
@assert size(V) == (n, k)
R = kmeans(V', k; maxiter=200, display=:final)
assignment_spectral = assignments(R)
after = Dates.now()
elapsedmin_spectral = round(((after - before) / Millisecond(1000))/60, digits=5) + Vhat_timecost 


# 5. evaluate clustering results using some metric 
A = getBodyWeightedAdj()
conduct = conductance(A, assignment)
conduct_spectral = conductance(A, assignment_spectral)

# 6. write results into results.txt
io = open("Reddit_results.txt", "a")
write(io, "\n$(Dates.now()), randseed: N/A \n" )
write(io, "Data set: RedditHyperlinks  testing points: $n; training data: $ntrain\n") 
write(io, "Target number of clusters: $k \n") 
write(io, "baseline: random \n")
write(io, "Time cost:       $(@sprintf("%.5f", elapsedmin))
conductance       $(@sprintf("%.5f", conduct)) \n")
write(io, "baseline: spectral clustering without attribute info \n")
write(io, "Time cost:       $(@sprintf("%.5f", elapsedmin_spectral))
conductance       $(@sprintf("%.5f", conduct_spectral)) \n")
close(io)

