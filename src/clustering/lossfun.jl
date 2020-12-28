"""
Computes geometric clustering loss function exactly using training data and
    in particular, LinkConstraintsMatrix, in addition to derivative
    for gradient-based minimization.
Inputs:
    - V: matrix of eigenvectors of Laplacian stacked column-wise
    - dV: derivative of V w.r.t theta
    - d: dimension of training points
    - theta: Laplacian hyperparameter
Outputs:
    - loss
    - dloss
"""
function loss_fun(X::Array{T, 2}, k, d, θ,
    traindata::AbstractTrainingData; normalized = true) where T<:Float64

    ntrain = traindata.n
    LinkConstraintsMatrix = traindata.LinkConstraintsMatrix
    (L, dL) = laplacian_L(X, θ)
    (V, Λ) = compute_eigs(Matrix(L), k)
    dV = comp_dV(V, Λ, L, dL, d; normalized = normalized)
    if normalized
        rownorms = mapslices(norm, V; dims = 2)
        V = broadcast(/, V, rownorms)
    end
    n = size(V, 1)
    dimθ = length(θ)
    @assert size(LinkConstraintsMatrix) == (ntrain, ntrain)
    if size(V) != (n, k)
        a = size(V, 1)
        b = size(V, 2)
        @show L[1:2, 1:2]
        @show θ
        error("size of V is $a by $b but should be $n by $k")
    end
    V_train = V[1:ntrain, :]
    #compute derivative
    dV_train = reshape(dV, n, k, dimθ)[1:ntrain, :, :]
    # derivative
    K1 = broadcast(-, reshape(V_train, ntrain, 1, k, 1), reshape(V_train, 1, ntrain, k, 1))
    K2 = broadcast(-, reshape(dV_train, ntrain, 1, k, dimθ), reshape(dV_train, 1, ntrain, k, dimθ))
    @assert size(K1) == (ntrain, ntrain, k, 1)
    @assert size(K2) == (ntrain, ntrain, k, dimθ)
    K3 = dropdims(sum(broadcast(*, K1, K2); dims=3); dims=3)
    @assert size(K3) == (ntrain, ntrain, dimθ)
    dloss = broadcast(*, reshape(LinkConstraintsMatrix, ntrain, ntrain, 1), K3)
    dloss = reshape(sum(dloss; dims=[1, 2]), dimθ)
    dloss = dimθ == 1 ? dloss[1] : dloss
    K = Array{Float64, 2}(undef, ntrain, ntrain)
    K = pairwise!(K, SqEuclidean(), V_train, dims=1)
    loss = dot(LinkConstraintsMatrix, K) ./ 2
    return loss, dloss
end
