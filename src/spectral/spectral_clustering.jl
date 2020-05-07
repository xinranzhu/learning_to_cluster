using Optim

include("../kernels/kernels.jl")
include("comp_mat_deriv.jl")

# train an optimal θ
function spectral_clustering_main(Xtest, X, y, k)
    n, d = size(X)
    ntest = size(Xtest)
    # generate constraints matrix Apm 
    Apm = gen_constraints(X, y)
    # optimize loss fun
    loss(θ) = loss_fun(θ, X, Apm, k)[1]
    loss_deriv!(G, θ) = loss_fun(θ, X, Apm, k, G)[2]
    results = optimize((loss, loss_deriv!, θ0))
    θ = Optim.minimum(results)
    # Compute eigenvectors on Xtest using the optimal θ 
    Vtest = spectral_clustering_model(Xtest, k, θ)
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


function gen_constraints(X, y)
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
