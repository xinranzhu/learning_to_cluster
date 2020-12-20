# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain


# using Optim
using Distances
using Dates
using Arpack
using LinearAlgebra
using KrylovKit
using LineSearches

include("../datastructs.jl")

"""
k: number of clusters
X:
θ: either initial value of theta
Xtrain:
ytrain: 
"""
function spectral_reduction_main(n::Int64, θ::Union{Array{T, 1}, Nothing},
                                traindata::Union{AttributedTrainingData, Nothing},
                                Vhat_set::NamedTuple) where T<:Float64
    # unpack Vhat 
    Vhat = Vhat_set.Vhat
    k = Vhat_set.k
    rangeθ = Vhat_set.rangeθ
    dimθ = size(rangeθ, 1)
    m = size(Vhat, 2)
    @assert m > k 
    # train an optimal θ value if have training set
    if traindata != nothing 
        before = Dates.now()
        @info "Start training θ"
        loss(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat; if_deriv = false)[1] 
        loss_deriv(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat)[2] 
        function loss_deriv!(G, θ)
            G .= loss_deriv(θ)
        end
        @info "Start training"
        # θ_init = rand(Uniform(rangeθ[1, 1], rangeθ[1, 2]), dimθ, 1)
        θ_init = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]

        # inner_optimizer = LBFGS()
        # inner_optimizer = ConjugateGradient()
        nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=0.1,scaled=true),
                           linesearch=LineSearches.Static())
        inner_optimizer = OACCEL(nlprecon=nlprecon, wmax=10)
        results = Optim.optimize(loss, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer))
        θ = Optim.minimizer(results)
        @info "Finish training, optimal θ" θ
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        @info "Trained θ, time cost " θ, elapsedmin
    end

    # check if θ is provided or trained
    if θ == nothing && traindata == nothing
        θ = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
        @warn "No training info or theta value provided, use random theta within range"
    end

    before = Dates.now()
    # compute H 
    L, _ = laplacian_attributed_L(θ; if_deriv = false)
    # PS: computing H takes 
    H = Vhat' * L * Vhat 
    @assert size(H) == (m, m) 

    # compute Y, k largest eigenvectors of H
    ef = eigen(Symmetric(H), m-k+1:m)
    Y = ef.vectors
    @assert size(Y) == (m, k)

    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    @info "H and Y computation time cost", elapsedmin

    # put Vhat*Y into kmeans
    @info "Start kmeans"
    @time a =  kmeans_reduction(Vhat, Y, k; maxiter = 200)
    # loss_opt = loss_fun_reduction(n, k, θ, traindata, Vhat; if_deriv = false)[1] 
    return a, θ
end

function comp_Vhat(n::Int64, k::Int64, rangeθ::Array{T, 2}; N_sample::Int = 100, precision::T = 0.995, debug::Bool = false) where T<:Float64
    # before = Dates.now()
    # n, d = size(X)
    dimθ = size(rangeθ, 1)
    @assert size(rangeθ, 2) == 2 
    # adjust N_sample if too large
    while (n > 10000) && (k*N_sample > 10000)
        @warn "To expensive to do svd Vhat_sample of ($n, $(k*N_sample))."
        N_sample = Int(floor(N_sample*0.8))
    end
    # use quasi-random samples
    s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
    N = hcat([next!(s) for i = 1:N_sample]...)' # N_sample * d
    Vhat_sample = Array{Float64, 2}(undef, n, k*N_sample)
    for i in 1:N_sample
        θ = N[i, :]
        L, _ = laplacian_attributed_L(θ; if_deriv = false) # 43s
        _, Vk = eigsolve(L, k, :LR, Float64; issymmetric=true, tol = 1e-16)
        Vhat_sample[:, (i-1)*k+1: i*k] = hcat(Vk...)
    end
    # compute Vhat from truncated SVD from LinearAlgebra
    F = svd(Vhat_sample)
    S = F.S # singular value
    totalsum = sum(S)
    partialsum = [S[1]]
    for i in 2:length(S)
        partialsum = append!(partialsum, partialsum[i-1] + S[i])
        if partialsum[i] > precision * totalsum
            @info "$i, partial sum = $(partialsum[i]),  total sum = $totalsum"
            break
        end
    end 
    m = max(length(partialsum), k)
    Vhat =  F.U[:, 1:m] # n by m
    @info "Finish computing Vhat , m = $m"
    return Vhat, S[m+1]
end

function loss_fun_reduction(n::Int64, k::Int64, θ::Union{Array{T, 1}, T}, traindata::AttributedTrainingData, Vhat::Array{T, 2}; 
                            if_deriv::Bool = true) where T<:Float64
    dimθ = length(θ)
    n, m = size(Vhat)
    ntrain = traindata.n
    idtrain = traindata.id
    Apm = traindata.Apm
    # compute Y(θ)
    L, dL = laplacian_attributed_L(θ; if_deriv = if_deriv) # takes 45s
    H = Vhat' * L * Vhat # takes 0.6s, m = 500
    @assert size(H) == (m, m) 
    # ef = eigen(Symmetric(H), m-k+1:m)
    # Y = ef.vectors
    # Λ = ef.values
    Y = eigvecs(Symmetric(H))[:, m-k+1:m]
    Λ = eigvals(Symmetric(H))[m-k+1:m]
    @assert size(Y) == (m, k)
    @assert length(Λ) == k
    # select training indices 
    Vhat_train_Y = (@view Vhat[idtrain, :]) * Y
    # compute loss
    K = Array{Float64, 2}(undef, ntrain, ntrain)
    K = pairwise!(K, SqEuclidean(), Vhat_train_Y, dims=1)
    loss = dot(Apm, K) ./ 2
    if if_deriv
        # compute d(Y(θ))
        dH = Array{Float64, 3}(undef, m, m, dimθ)
        for i in 1:dimθ
            dH[:,:,i] = Vhat' * dL[i] * Vhat
        end
        dY = comp_dY(Y, Λ, H, dH, dimθ)
        # compute dloss
        dloss_test = Array{Float64, 1}(undef, dimθ)
        for i in 1:dimθ
            Vhat_train_dY = @views Vhat[idtrain, :] * dY[:, :, i]
            K = pairwise!(K, SqEuclidean(), Vhat_train_Y, Vhat_train_dY, dims=1)
            dloss_test[i] = dot(Apm, K)
        end
        dloss_test = reshape(dloss_test, dimθ)
        # compute dloss_test
        Vhat_train_dY = Array{Float64, 3}(undef, ntrain, k, dimθ)
        Vhattrain = Vhat[idtrain, :]
        @tensor Vhat_train_dY[i, j, k] = Vhattrain[i, s] * dY[s, j, k]
        K1 = broadcast(-, reshape(Vhat_train_Y, ntrain, 1, k, 1), reshape(Vhat_train_Y, 1, ntrain, k, 1))
        K2 = broadcast(-, reshape(Vhat_train_dY, ntrain, 1, k, dimθ), reshape(Vhat_train_dY, 1, ntrain, k, dimθ))
        @assert size(K1) == (ntrain, ntrain, k, 1)
        @assert size(K2) == (ntrain, ntrain, k, dimθ)
        K3 = dropdims(sum(broadcast(*, K1, K2); dims=3); dims=3)
        @assert size(K3) == (ntrain, ntrain, dimθ)
        dloss_test = broadcast(*, reshape(Apm, ntrain, ntrain, 1), K3)
        dloss_test = reshape(sum(dloss_test; dims=[1, 2]), dimθ)
        dloss_test = dimθ == 1 ? dloss_test[1] : dloss_test
        
        dloss = zeros(dimθ)

        R = CartesianIndices(Apm)
        Vhattrain = @view Vhat[idtrain, :]
        dloss = zeros(dimθ)
            for i in 1:dimθ
            YdY = @views Y[:, :] * dY[:, :, i]'
            @info size(i)
            for I in R
                j = Tuple(I)[1]
                k = Tuple(I)[2]
                # @info "i, j = $i, $j"
                wjk = Vhattrain[j,:] - Vhattrain[k,:]
                # @info size(wjk)
                dloss[i] += Apm[I] .* (wjk'*YdY*wjk)
            end
            compare_dloss[i] = abs(dloss_test[i] - dloss[i])
        end
    else 
        dloss = nothing
    end
    @info "Evaluate loss fun, current loss" loss, θ
    return loss, dloss
end




function comp_dY(Y::Array{T, 2}, Λ::Array{T, 1}, H::Array{T, 2}, dH::Union{Array{T, 2}, Array{T, 3}}, dimθ::Int64) where T<:Float64
    m, k = size(Y)
    dY = Array{Float64, 3}(undef, m, k, dimθ)
    for i in 1:k
        y = Y[:, i]
        if dimθ == 1
            dHy = dH * y
        else
            dHy = Array{Float64, 2}(undef, m, dimθ)
            @tensor dHy[i, j] = dH[i, s, j] * y[s]
        end
        dHy .-= y * (y' * dHy)
        dY[:, i, :] = [(Λ[i] * I(m) - H); y'] \ [dHy; zeros(dimθ)'] 
    end
    # dY = dimθ == 1 ? dropdims(dY; dims = 3) : dY
    return dY
end

function kmeans_reduction(Vhat::Array{T, 2}, Y::Array{T, 2}, k::Int; maxiter::Int = 200) where T<:Float64
    R = kmeans((Vhat*Y)', k; maxiter=maxiter, display=:final)
    @assert nclusters(R) == k # verify the number of clusters
    a = assignments(R) # get the assignments of points to clusters
    return a
end 