# using Optim
include("../kernels/kernels.jl")
include("comp_deriv.jl")
include("../datastructs.jl")

# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain
function spectral_clustering_main(X::Array{T, 2}, k::Int, traindata::AbstractTrainingData, rangeθ::Array{T, 2}) where T<:Float64
    dimθ = size(rangeθ, 1)
    ntrain = traindata.n
    Apm = traindata.Apm
    ntotal, d = size(X)
    # optimize loss fun
    loss(θ) = loss_fun(X, k, θ, traindata; if_deriv = false)[1]
    loss_deriv(θ) = loss_fun(X, k, θ, traindata)[2]
    function loss_deriv!(G, θ)
        G = loss_fun(X, k, θ, traindata)[2]
    end
    @info "Start training"
    if dimθ == 1
        results = Optim.optimize(loss, rangeθ[1, 1], rangeθ[1, 2])
        θ = Optim.minimizer(results)
    else
        θ_init = rand(Uniform(rangeθ[1, 1], rangeθ[1, 2]), dimθ, 1)
        inner_optimizer = LBFGS()
        results = Optim.optimize(loss, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer))
        θ = Optim.minimizer(results)
    end
    @info "Finish training, optimal θ" θ
    # Compute eigenvectors on Xtest using the optimal θ
    V, _ = spectral_clustering_model(X, k, θ; if_deriv = false)
    return V, θ
end


function spectral_clustering_model(X::Array{T, 2}, k::Int, θ::Union{Array{T, 1}, T}; if_deriv::Bool = true) where T<:Float64
    # @info "enter spectral clustering model"
    n, d = size(X)
    dimθ = length(θ)
    # generate the Laplacian matrix
    L, dL = laplacian_L(X, θ; if_deriv = if_deriv)
    @info "size(L) and current θ " size(L), θ
    # @info "finish computing L, dL" size(L), size(dL)
    # compute k eigenvectors, store in V
    ef = eigen(Symmetric(L), n-k+1:n)
    V = ef.vectors
    Λ = ef.values
    # @info "finish computing V" size(V), n, k, size(ef.vectors)
    @assert size(V, 2) == k # make sure returns k eigenvectors
    # @info "finish computing V" size(V)

    # compute dV
    dV = if_deriv ? comp_dV_L(V, Λ, L, dL, dimθ) : nothing
    # @info "finish computing dV"

    # normalize rows of V
    # rownorms = mapsliceås(norm, V; dims = 2)
    # V = broadcast(/, V, rownorms)
    return V, dV
end

function loss_fun(X::Array{T, 2}, k::Int, θ::Union{Array{T, 1}, T}, traindata::AbstractTrainingData; if_deriv::Bool = true) where T<:Float64
    ntrain = traindata.n
    Apm = traindata.Apm
    n, d = size(X)
    dimθ = length(θ)
    @assert size(Apm) == (ntrain, ntrain)
    # compute clustering
    V, dV = spectral_clustering_model(X, k, θ)
    @assert size(V) == (n, k)
    V_train = V[1:ntrain, :]
    if if_deriv
        dV_train = reshape(dV, n, k, dimθ)[1:ntrain, :, :]
        # derivative
        K1 = broadcast(-, reshape(V_train, ntrain, 1, k, 1), reshape(V_train, 1, ntrain, k, 1))
        K2 = broadcast(-, reshape(dV_train, ntrain, 1, k, dimθ), reshape(dV_train, 1, ntrain, k, dimθ))
        @assert size(K1) == (ntrain, ntrain, k, 1)
        @assert size(K2) == (ntrain, ntrain, k, dimθ)
        K3 = dropdims(sum(broadcast(*, K1, K2); dims=3); dims=3)
        @assert size(K3) == (ntrain, ntrain, dimθ)
        dloss = broadcast(*, reshape(Apm, ntrain, ntrain, 1), K3)
        dloss = reshape(sum(dloss; dims=[1, 2]), dimθ)
        dloss = dimθ == 1 ? dloss[1] : dloss
    else
        dloss = nothing
    end
    K = Array{Float64, 2}(undef, ntrain, ntrain)
    K = pairwise!(K, SqEuclidean(), V_train, dims=1)
    loss = dot(Apm, K) ./ 2
    return loss, dloss
end

"""
Compute Derivative of Eigenvectors w.r.t hyperparameters
"""
function comp_dV_L(V::Array{T, 2}, Λ::Array{T, 1}, L::Symmetric{T,Array{T,2}}, dL::Union{Array{T, 2}, Array{T, 3}}, dimθ::Int64) where T<:Float64
    n, k = size(V)
    dV = Array{Float64, 3}(undef, n, k, dimθ)
    for i in 1:k
        v = V[:, i]
        if dimθ == 1
            dLv = dL * v
        else
            dLv = dHy = Array{Float64, 2}(undef, n, dimθ)
            @tensor dLv[i, j] = dL[i, s, j] * v[s]
        end
        dLv .-= v * (v' * dLv)
        dV[:, i, :] =  [(Λ[i] * I(n) - L); v'] \ [dLv; zeros(dimθ)'] #solve linear system for derivative of eigenvector
    end
    dV = dimθ == 1 ? dropdims(dV; dims = 3) : dV
    return dV
end
