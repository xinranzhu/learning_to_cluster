"""
Compute k leading eigenvectors and eigenvalues of L
"""
function compute_eigs(L::Array{T, 2}, k::Int) where T<:Float64
    n = size(L,1)
    # compute k eigenvectors, store in V
    #ef = eigen(Symmetric(L), n-k+1:n)
    ef = eigen(Symmetric(L), n-k+1:n)
    V = ef.vectors
    Λ = ef.values
    return V, Λ
end

"""
Computes clustering assignment using classical spectral clustering.
    L: Laplacian matrix
    k: number of clusters
"""
function cluster_spectral(L::Array{T, 2}, k::Int; normalized = true) where T<:Float64
    V, Λ = compute_eigs(L, k)
    # @info "finish computing V" size(V), n, k, size(ef.vectors)
    @assert size(V, 2) == k # make sure returns k eigenvectors
    # normalize rows of V
    if normalized
        rownorms = mapslices(norm, V; dims = 2)
        V = broadcast(/, V, rownorms)
    end
    R = kmeans(V', k; maxiter=100, display=:final)
    assignment = assignments(R)
    return assignment
    #return V#, dV
end
