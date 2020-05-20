using SparseArrays
"""
A is (possibly weighted) adjacency matrix of graph, y is assignment of vertices to clusters
we assume the clusters are numbered 1, 2, ..., k.

Works for both SparseArray and regular dense array
"""
function conductance(A, y::Array{T, 1} where T<:Int64)
    #println(typeof(y))
    n = size(A, 1)
    k = maximum(y)
    #println(k)
    conductances = zeros(1, Int64(k))
    clusters = [Array{Int64, 1}(undef, 0) for i = 1:k] #clusters[i] contains indices of nodes contained in cluster i
    for i = 1:n
       push!(clusters[y[i]], i) 
    end   
    total_edge_weights = sum(A)
    for i = 1:k 
        denom1 = sum(A[clusters[i], :])
        denom2 = total_edge_weights - denom1
        #println(denom1)
        #println(denom2)
        denom = min(denom1, denom2)
        B = collect(1:n) 
        deleteat!(B, clusters[i]) #form complement to clusters[i]
        numer = sum(A[clusters[i], B])
        conductances[i] =  numer/denom
    end
    return sum(conductances)/k
end
# A = sprand(10, 10, .5)
# println(conductance(A, vcat(ones(Int64, 5), 2*ones(Int64, 5))))
# B = ones(5, 5)
# C = zeros(5, 5)
# D = [B C; C B]
# println(conductance(D, vcat(ones(Int64, 5), 2*ones(Int64, 5))))
