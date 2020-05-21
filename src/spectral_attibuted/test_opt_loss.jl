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
using Optim
using Arpack
using KrylovKit

include("../attributed/attributed.jl")
include("../datastructs.jl")
include("model_reduction.jl")

n = 35776

(idtrain, ytrain) = trainInfo_fixed()
ntrain = length(idtrain)
traindata = atttraindata(ntrain, idtrain, ytrain)
@info "Size of training data" ntrain

VV = JLD.load("./saved_data/Vhat_set_1_1_1.jld")["data"]
Vhat = VV.Vhat
m = size(Vhat, 2)
k = VV.k
rangeθ = VV.rangeθ
dimθ = size(rangeθ, 1)
N_sample = VV.N_sample
@info "Finish loading Vhat, k = $k, N_sample = $N_sample"

θ_init = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
@info "θ_init = $θ_init."

L, dL = laplacian_attributed_L(θ_init)

loss(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat; if_deriv = false)[1] 
loss_deriv(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat)[2] 
function loss_deriv!(g, θ)
    g .= loss_deriv(θ)
end

# using Optim
before = Dates.now()
inner_optimizer = ConjugateGradient()

# results = Optim.optimize(loss, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, LBFGS())

# results = Optim.optimize(loss, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, GradientDescent())

# nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
#                            linesearch=LineSearches.Static())
# inner_optimizer = OACCEL(nlprecon=nlprecon, wmax=10)
results = Optim.optimize(loss, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer))

θ = Optim.minimizer(results)
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

@info "Finish optimizing, optimal θ = $θ."

io = open("Test_opt_loss.txt", "a")
write(io, "\n$(Dates.now()), randseed: N/A \n" )
write(io, "Data set: RedditHyperlinks  testing points: $n; training data: $ntrain\n") 
write(io, "k: $k;   N_sample: $N_sample;   m: $m \n")
write(io, " Time cost:  $(@sprintf("%.5f", elapsedmin))
            optimal θ = $θ \n")
close(io)

