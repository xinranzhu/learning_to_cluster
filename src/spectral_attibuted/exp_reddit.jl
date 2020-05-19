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
include("../attributed/clusteringAttributed.jl")


s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--ntotal"
        help = "size of testing set"
        arg_type = Int
        default = 35776
    "--k"
        help = "target number of clustering"
        arg_type = Int
        default = 15
    "--set_range"
        help = "use which settings of rangeθ -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--set_Nsample"
        help = "use which settings of N_sample -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--idtheta"
        help = "assigned value of theta, skip training -- should be load from file"
        arg_type = Int
        default = 0
end
parsed_args = parse_args(ARGS, s)

# 1. load data or setup the following
n = parsed_args["ntotal"]
k = parsed_args["k"]
dimθ = 12
range_set = JLD.load("./saved_data/reddit_range_set.jld")["data"]
rangeθ = reshape(range_set[parse_args["set_range"], :, :], 12, 2)

θ = nothing; atttraindata = nothing
if parse_args["idtheta"] == 0 # 2a) training data/constraints
    # TODO: set Apm::Symmetric{Int64,Array{Int64,2}} (PS: but can be other form like must-link node indices Ap and cannot-link node indices Am)
    # TODO: set training label y::Array{Int64, 1}
    @assert size(Apm, 1) == length(y)
    ntrain = size(Apm, 1)
    atttraindata = atttraindata(ntrain, y, Apm)
    ntrain = traindata.n
    @info "Size of training data" ntrain
else # 2b) skip training, load some assigned theta
    θ_set = JLD.load("./saved_data/reddit_theta_set.jld")["data"] # N * 12
    @assert size(θ_set, 2) == dimθ
    θ = θ_set[parse_args["idtheta"], :]
    @assert (θ .> rangeθ[:, 1]) && (θ .< rangeθ[:, 2])
end

# 3. load Vhat
Vhat_set = JLD.load("./saved_data/Vhat_set_$(parsed_args["set_range"])_$(parsed_args["set_Nsample"]).jld")["data"]
N_sample = Vhat_set.N_sample
Vhat_timecost = Vhat_set.timecost
m = size(Vhat_set.Vhat, 2)
@assert Vhat_set.rangeθ == rangeθ 
@info "Finish loading Vhat, N_sample, m, timecost" N_sample, m, Vhat_timecost


# 4. clustering 
before = Dates.now()
assignment, θ = spectral_reduction_main(n, k, rangeθ, θ, atttraindata, Vhat_set) 
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) + Vhat_timecost 

# 5. evaluate clustering results using some metric 
#TODO

# 6. write results into results.txt
io = open("Reddit_results.txt", "a")
write(io, "\n$(Dates.now()), randseed: N/A \n" )
write(io, "Data set: RedditHyperlinks  testing points: $n; training data: $ntrain\n") 
write(io, "rangeθ_id: $(parse_args["set_range"]),  dimθ: $dimθ;   N_sample: $N_sample;    m: $m \n")
write(io, " 
    Time cost:   $(@sprintf("%.5f", elapsedmin))
    metric1      $(@sprintf("%.5f", max_acc)) \n")
close(io)

