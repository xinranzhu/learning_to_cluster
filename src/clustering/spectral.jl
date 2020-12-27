
"""
L: Laplacian matrix
k: number of clusters
"""
function cluster_spectral(L::Array{T, 2}, k::Int) where T<:Float64
    n = size(L,1)
    # compute k eigenvectors, store in V
    #ef = eigen(Symmetric(L), n-k+1:n)
    ef = eigen(Symmetric(L), n-k+1:n)
    V = ef.vectors
    Λ = ef.values
    # @info "finish computing V" size(V), n, k, size(ef.vectors)
    @assert size(V, 2) == k # make sure returns k eigenvectors
    # @info "finish computing V" size(V)
    # compute dV
    #dV = derivative ? comp_dV_L(V, Λ, L, dL, dimθ) : nothing
    # @info "finish computing dV"
    # normalize rows of V
    rownorms = mapslices(norm, V; dims = 2)
    V = broadcast(/, V, rownorms)
    R = kmeans(V', k; maxiter=100, display=:final)
    assignment = assignments(R)
    return assignment
    #return V#, dV
end
