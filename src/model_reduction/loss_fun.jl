"""
    loss_fun(X::Array{T, 2}, k::Int64, θ::Union{Array{T, 1}, T}, traindata::AbstractTrainingData, V̂::Array{T, 2}[,
            I_rows::Union{Array{Int64,1}, Nothing}=nothing, if_deriv::Bool=true])

Computes the clustering loss between the true and reduced spectral clustering.

Arguments:
- `X::Array{Float64, 2}`: original dataset to cluster.
- `k::Int64`: parameter for the dimensionality of the invariant subspace desired
- `θ::Union{Array{T, 1}, T}`: sampling parameters.
- `trainingData::AbstractTrainingData`: subset of data with ground truth labels
- `V̂::Array{T, 2}`: approximation space.
- `I_rows::Union{Array{Int64,1}, Nothing}`: the rows selected during model reduction.
- `if_deriv::Bool`: if the derivative of the loss should be used.

Returns:
- `loss`: pairwise loss.
- `dloss`: derivative pairwise loss.
"""
function loss_fun(X::Array{T, 2}, k::Int64, θ::Union{Array{T, 1}, T}, traindata::AbstractTrainingData, V̂::Array{T, 2};
                  I_rows::Union{Array{Int64,1}, Nothing}=nothing, if_deriv::Bool=true) where T<:Float64
    dimθ = length(θ)
    n, m = size(V̂)
    ntrain = traindata.n
    idtrain = 1:ntrain
    Apm = traindata.Apm
    # compute Y(θ)
    L, dL = laplacian_L(X, θ; if_deriv = if_deriv) # 9s
    # H = I_rows == nothing ?  Vhat' * L * Vhat : (Vhat[I_rows, :])' * (L[I_rows, ] * Vhat) # 70s if m = 785
    H = @views V̂'[:, :] * L[:, :] * V̂[:, :] # 20s using @views
    @assert size(H) == (m, m)
    ef = eigen(Symmetric(H), m-k+1:m) # 0.06s, m = 785
    Y = ef.vectors
    Λ = ef.values
    # select training indices
    V̂_train_Y = (@view V̂[1:ntrain, :]) * Y
    # compute loss
    K = Array{Float64, 2}(undef, ntrain, ntrain)
    K = pairwise!(K, SqEuclidean(), V̂_train_Y, dims=1)
    loss = dot(Apm, K) ./ 2
    if if_deriv
        # compute d(Y(θ))
        if dimθ == 1
            # dH = I_rows == nothing ? Vhat' * dL * Vhat : (Vhat[I_rows, :])' * (dL[I_rows, ] * Vhat)
            # dH = Vhat' * dL * Vhat
            dH = @views V̂'[:, :] * dL[:, :] * V̂[:, :]
        else
            dH = Array{Float64, 3}(undef, m, m, dimθ)
            @tensor dH[i,j,k] = V̂'[i, s] * dL[s, l, k] * V̂[l, j]
        end
        dY = comp_dY(Y, Λ, H, dH, dimθ)
        if dimθ > 1
            V̂_train_dY = Array{Float64, 3}(undef, ntrain, k, dimθ)
            V̂_train = @view V̂[idtrain, :]
            @tensor V̂_train_dY[i, j, k] = V̂_train[i, s] * dY[s, j, k]
        else
            V̂_train_dY = (@view V̂[idtrain, :]) * dY
        end

        # Compute the must-link (K1) and must-not-link (K2) loss, then output.
        K1 = broadcast(-, reshape(V̂_train_Y, ntrain, 1, k, 1), reshape(V̂_train_Y, 1, ntrain, k, 1))
        K2 = broadcast(-, reshape(V̂_train_dY, ntrain, 1, k, dimθ), reshape(V̂_train_dY, 1, ntrain, k, dimθ))
        @assert size(K1) == (ntrain, ntrain, k, 1)
        @assert size(K2) == (ntrain, ntrain, k, dimθ)
        K3 = dropdims(sum(broadcast(*, K1, K2); dims=3); dims=3)
        @assert size(K3) == (ntrain, ntrain, dimθ)
        dloss = broadcast(*, reshape(Apm, ntrain, ntrain, 1), K3)
        dloss = reshape(sum(dloss; dims=[1, 2]), dimθ)
        dloss = dimθ == 1 ? dloss[1] : dloss
        @info "Evaluate loss func, θ and loss" θ, loss, norm(dloss)
    else
        dloss = nothing
    end
    return loss, dloss
end
