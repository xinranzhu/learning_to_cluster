using CSV
using DataFrames
using Clustering
using Hungarian
using Combinatorics
using Statistics
using LinearAlgebra
using Test
include("../../src/clustering/spectral.jl")
include("../../src/clustering/laplacian.jl")

###########################################
######## DATA LOAD AND PROCESSING #########
###########################################
# load abalone
pwd()
df = DataFrame(CSV.File("experiments/datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8])
label = convert(Array, df[:, 9]) # 1 ~ 29
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
L = Matrix(laplacian_L(data, 1.0*ones(7))[1])
sc_assignment = cluster_spectral(L, 6)
sc_max_acc, matched_assignment = bipartite_match_labels(sc_assignment, label, 6)
@test sc_max_acc > max_acc
