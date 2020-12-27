using Clustering
using Hungarian
using Combinatorics
using Statistics
using Test
include("../../src/clustering/kmeans_match_labels.jl")

## test kmeans and bipartite label matching

# choose points in 3-space that are easily classified
X = [1 1.0 0.0; -1.0 1.0 0.0; -1.0 1.5 0.0]
R = kmeans(X', 2; maxiter=20, display=:final)
@assert nclusters(R) == 2 # verify the number of clusters
assignment = assignments(R) # get the assignments of points to clusters
assignment

#observation:
# - true labels have to be subset of {1, 2, 3..., k}
y = [2, 1, 1]
trainmax = 0
matched_assignment
max_acc, matched_assignment = bipartite_match_labels(assignment, y, 2) # assignment is updated
RI = randindex(matched_assignment[trainmax+1:end], y[trainmax+1:end])

max_acc

@test max_acc == 1
