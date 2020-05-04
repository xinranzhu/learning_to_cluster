# Given n data points, given the similarity parameter
# do spectral clustering
# θ can be picked to be fixed or trained from training set
include("../kernels/kernels.jl")

function spectral_clustering(X, k, θ)
    n, d = size(X)
    # generate the affinity matrix
    if length(θ)>1 # θ = 1/sigma^2
        A = correlation(Gaussian(), θ, X; jitter = 0) 
    else
        A = correlation(Gaussian(), θ[1], X; jitter = 0) 
    end    
    A = A .- Diagonal(diag(A)) # A_ii = 0
    # Define degree matrix
    D = reshape(sum(A, dims = 2), n)
    # normalize the laplacian matrix 
    sqrtD = 1 ./ sqrt.(D)
    L = Diagonal(sqrtD) * A * Diagonal(sqrtD);
    # compute k eigenvectors, store in V
    V = eigvecs(L)[:, n-k+1:end]
    @assert size(V, 2) == k # make sure returns k eigenvectors
    # normalize rows of V
    # rownorms = mapslices(norm, V; dims = 2)
    # V = broadcast(/, V, rownorms)
    return V 
end

function loss_fun(θ, X, Apm, k)
    # compute clustering
    V = spectral_clustering(X, k, θ)
    R = CartesianIndices(Apm)
    loss = 0.
    for I in R 
        global loss
        i, j = Tuple(I) # get current node pair
        dij = V[i, :] - V[j, :]
        loss += Apm[I] * dot(dij, dij)
    end

    # derivative
    
end


function gen_constraints(X, label)
    n, d = size(X)
    Apm = Array{Float64, 2}(undef, n, n)
    R = CartesianIndices(Apm)
    for I in R 
        i, j = Tuple(I)
        if i == j
            constraint = 0
        elseif y[i] == y[j]
            constraint = 1
        else
            constraint = -1
        end
        Apm[I] = constraint
    end
    return Apm
end