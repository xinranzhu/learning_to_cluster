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

    # r1 N1 k1
    # θ = [9.487621844447142, 0.09842062529603623, 0.06160502411393698, 0.6062265189824965, 0.641008699734444, 0.08294606440044738, 0.48870926912866786, 0.6034674948811787, 0.6344423727195974, 0.5608781057446472, 0.19902204556904857, 0.36937683445247593]
    # rid = 1; Nid = 1; kid = 1

    # r1 N1 k2
    # θ = [8.280493605683557, 0.1488694463561837, 0.9321122474273189, 0.6877090070436545, 0.18100136197534922, 0.2693995586303868, 0.7611397864324396, 0.7732606535275299, 0.05498093130890114, 0.01806525028568884, 0.9154320265107143, 0.40243896373698407]
    # rid = 1; Nid = 1; kid = 2

    # r1 N1 k4
    θ = [2.7545352975365907, 0.2225150550912589, 0.5406648005644531, 0.16745922043883968, 0.05404266869254614, 0.5275948384428386, 0.043270972016568904, 0.8825064906796511, 0.15515554391790015, 0.2846260929550612, 0.22027616069036676, 0.5532294408875795]
    rid = 1; Nid = 1; kid = 3

    # r1 N2 k1
    # θ = [2.6671381734158066, 0.5931582086561903, 0.5001246527165628, 0.6134644474696354, 0.8371539317263909, 0.1359449913444988, 0.6870162040618238, 0.78957836832319, 0.6271068207560877, 0.8419196053267872, 0.5010582532123371, 0.7346727483982489]
    # rid = 1; Nid = 2; kid = 1

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
# @assert sum(θ .> rangeθ[:, 1]) + sum(θ .< rangeθ[:, 2]) == 2*dimθ
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

