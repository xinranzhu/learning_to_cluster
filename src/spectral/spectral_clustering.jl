using Optim

include("../kernels/kernels.jl")
include("comp_mat_deriv.jl")

# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain
function spectral_clustering_main(X, Xtrain, ytrain, k, dimθ)
    ntrain, dtrain= size(Xtrain)
    ntotal = size(X)
    # generate constraints matrix Apm 
    Apm = gen_constraints(Xtrain, ytrain)
    # optimize loss fun
    loss(θ) = loss_fun(θ, Xtrain, Apm, k)[1] #TODO
    loss_deriv!(G, θ) = loss_fun(θ, Xtrain, Apm, k, G)[2] #TODO
    θ_init = rand(dimθ)
    results = optimize((loss, loss_deriv!, θ_init))
    θ = Optim.minimum(results)
    # Compute eigenvectors on Xtest using the optimal θ 
    Vtest = spectral_clustering_model(X, k, θ)
end


function spectral_clustering_model(X, k, θ)
    n, d = size(X)
    # generate the affinity matrix
    A = affinity_A(X, θ)[1]
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

function loss_fun(θ, X, Apm, k, loss_deriv)
    # compute clustering
    V = spectral_clustering(X, k, θ)
    R = CartesianIndices(Apm)
    loss = 0.
    for I in R 
        # global loss
        i, j = Tuple(I) # get current node pair
        dij = V[i, :] - V[j, :]
        loss += Apm[I] * dot(dij, dij)
    end
    # derivative
    loss_deriv = nothing
end


