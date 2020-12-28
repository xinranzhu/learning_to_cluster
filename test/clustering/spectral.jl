using CSV
using DataFrames
using Clustering
using Hungarian
using Combinatorics
using Statistics
using LinearAlgebra
using Test
using LineSearches
using Optim
using Plots
include("../../src/core/datastructs.jl")
include("../../src/clustering/spectral.jl")
include("../../src/clustering/laplacian.jl")
include("../../src/clustering/lossfun.jl")
include("../../src/clustering/kmeans_match_labels.jl")

include("../../src/l2c.jl")
###########################################
######## DATA LOAD AND PROCESSING #########
###########################################
# load abalone
pwd()
df = DataFrame(CSV.File("experiments/datasets/abalone.csv", header = 0))
data = convert(Matrix, df[500:1500,2:8])
label = convert(Array, df[500:1500, 9]) # 1 ~ 29
k = 29
# relabel: regroup labels <= 5 as one lable, and >=15 as one label
# then target number of clusters = 11
label[label .<= 6] .= 1
label[(&).(label .> 6, label .<=8)] .= 2
label[(&).(label .> 8, label .<=10)] .= 3
label[(&).(label .> 10, label .<=12)] .= 4
label[(&).(label .> 12, label .<=14)] .= 5
label[label .> 14] .= 6
k = 6

@info "Target number of clusterings $k"
## K-means clustering
R = kmeans(data', 6; maxiter=100, display=:final)
@assert nclusters(R) == 6 # verify the number of clusters
assignment = assignments(R) # get the assignments of points to clusters

max_acc, matched_assignment = bipartite_match_labels(assignment, label, 6) # assignment is updated
#RI = randindex(matched_assignment[trainmax+1:end], y[trainmax+1:end])

## Spectral clustering with theta = ones(n)
L = Matrix(laplacian_L(data, ones(7))[1])
sc_assignment = cluster_spectral(L, 6)
sc_max_acc, matched_assignment = bipartite_match_labels(sc_assignment, label, 6)
@show sc_max_acc
#@test sc_max_acc > max_acc

function SC(theta; normalized = true)
    L = Matrix(laplacian_L(data, theta)[1])
    sc_assignment = cluster_spectral(L, 6; normalized = normalized)
    sc_max_acc, matched_assignment = bipartite_match_labels(sc_assignment, label, 6)
    return sc_max_acc
end

## Train for θ using labeled data and geometric loss function
k = 6
d = 7
mytrain = trainingData(data, label, 333)
q(θ) = loss_fun(data, k, d, θ, mytrain; normalized = true)[1]
dq(θ) = loss_fun(data, k, d, θ, mytrain; normalized = true)[2]
@time q(ones(d))
@time dq(ones(d))
rangeθ = hcat(0.01*reshape(ones(d), d, 1), 100*reshape(ones(d), d, 1))
function loss_deriv!(G, θ)
    G .= dq(θ)
end
θ_init = ones(d)

(L, dL) = laplacian_L(data, 0.001*ones(d))
A = affinity_A(data, 100*ones(d))
L
#derivative check passes
if false
    (r1, r2, r3, r4) = checkDerivative(q, dq, ones(d), nothing, 2, 13)
    @show r4
end
# inner_optimizer = GradientDescent(
#                                     alphaguess = LineSearches.InitialStatic(alpha = 2., scaled = false),
#                                     linesearch = LineSearches.StrongWolfe())
#nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha = 0.1, scaled = false),
    #                        linesearch=LineSearches.StrongWolfe())
#inner_optimizer = OACCEL(nlprecon=nlprecon, wmax=10)
inner_optimizer = LBFGS()
# inner_optimizer = ConjugateGradient()
results = Optim.optimize(q, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer), Optim.Options(show_trace=true, time_limit = 300.0))
optθ = Optim.minimizer(results)
@show optθ
## Trained optimal θ vs random initialization ones(d)
N = 50
avg = 0.0
for i = 1:N
    global avg += SC(optθ)
end
avg2 = 0.0
for i = 1:N
    global avg2 = avg2 + SC(ones(7))
end

@show avg/N
@show avg2/N

## normalized vs unnormalized SC
N = 50
avg = 0.0
for i = 1:N
    global avg += SC(ones(d), normalized = false)
end
avg2 = 0.0
for i = 1:N
    global avg2 = avg2 + SC(ones(d), normalized = true)
end

@show avg/N
@show avg2/N

## spectral clustering evaluation function from l2c.jl

all = Dict()

for i = 1:9
    frac_train = i * 0.1
    res = evaluate_spectral_clustering(data, label; frac_train = frac_train, train = true, ntrials = 20)
    all[frac_train] = res
end

@show all
