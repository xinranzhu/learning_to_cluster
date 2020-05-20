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

θ = nothing; traindata = nothing
if parse_args["idtheta"] == 0 # 2a) training data/constraints
    (idtrain, ytrain) = trainInfo_fixed()
    ntrain = length(idtrain)
    traindata = atttraindata(ntrain, idtrain, ytrain)
    @info "Size of training data" ntrain
else # 2b) skip training, load some assigned theta
    θ_set = JLD.load("./saved_data/reddit_theta_set.jld")["data"] # N * 12
    @assert size(θ_set, 2) == dimθ
    θ = θ_set[parse_args["idtheta"], :]
    @assert (θ .> rangeθ[:, 1]) && (θ .< rangeθ[:, 2])
end

# load Vhat
Vhat_set = JLD.load("./saved_data/Vhat_set_$(parsed_args["set_range"])_$(parsed_args["set_Nsample"])_$(parsed_args["set_k"]).jld")["data"]
k = Vhat_set.k
rangeθ = Vhat_set.rangeθ
N_sample = Vhat_set.N_sample
Vhat_timecost = Vhat_set.timecost
m = size(Vhat_set.Vhat, 2)
@info "Finish loading Vhat, N_sample, m, timecost" N_sample, m, Vhat_timecost


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
write(io, "rangeθ_id: $(parse_args["set_range"]),  dimθ: $dimθ;   N_sample: $N_sample;    m: $m \n")
write(io, " 
    Time cost:   $(@sprintf("%.5f", elapsedmin))
    metric1      $(@sprintf("%.5f", max_acc)) \n")
close(io)

