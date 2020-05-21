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
if parsed_args["idtheta"] == 0 # 2a) training data/constraints
    (idtrain, ytrain) = trainInfo_fixed()
    ntrain = length(idtrain)
    traindata = atttraindata(ntrain, idtrain, ytrain)
    @info "Size of training data" ntrain
else # 2b) skip training, load some assigned theta
    # θ_set = JLD.load("./saved_data/reddit_theta_set.jld")["data"] # N * 12
    # @assert size(θ_set, 2) == dimθ
    # θ = θ_set[parsed_args["idtheta"], :]
    θ = [9.487621844447142, 0.09842062529603623, 0.06160502411393698, 0.6062265189824965, 0.641008699734444, 0.08294606440044738, 0.48870926912866786, 0.6034674948811787, 0.6344423727195974, 0.5608781057446472, 0.19902204556904857, 0.36937683445247593]
    rid = 1; Nid = 1; kid = 1
    @info "rid = $rid, Nid = $Nid, kid = $kid"
end
# load Vhat
Vhat_set = JLD.load("./saved_data/Vhat_set_$(rid)_$(Nid)_$(kid).jld")["data"]
k = Vhat_set.k
rangeθ = Vhat_set.rangeθ
dimθ = size(rangeθ, 1)
N_sample = Vhat_set.N_sample
Vhat_timecost = Vhat_set.timecost
m = size(Vhat_set.Vhat, 2)
@assert sum(θ .> rangeθ[:, 1]) + sum(θ .< rangeθ[:, 2]) == 2*dimθ
@info "Finish loading Vhat, N_sample, m, k,timecost" N_sample, m, k, Vhat_timecost

# clustering 
before = Dates.now()
assignment, θ = spectral_reduction_main(n, θ, traindata, Vhat_set) 
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) + Vhat_timecost 

# 5. evaluate clustering results using some metric 
A = getBodyWeightedAdj()
conduct = conductance(A, assignment)

# 6. write results into results.txt
io = open("Reddit_results.txt", "a")
write(io, "\n$(Dates.now()), randseed: N/A \n" )
write(io, "Data set: RedditHyperlinks  testing points: $n; training data: $ntrain\n") 
write(io, "rangeθ_id: $rid,  dimθ: $dimθ;   N_sample: $N_sample;    m: $m \n")
write(io, "Target number of clusters: $k \n") 
write(io, "Optimal θ: $θ \n") 
write(io, "Time cost:       $(@sprintf("%.5f", elapsedmin))
conductance      $(@sprintf("%.5f", conduct)) \n")
close(io)

