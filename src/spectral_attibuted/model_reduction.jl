# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain


# using Optim
using Distances
using Dates
using Arpack
using LinearAlgebra

include("../datastructs.jl")

"""
k: number of clusters
X:
θ: either initial value of theta
Xtrain:
ytrain: 
"""
function spectral_reduction_main(n::Int, k::Int, θ::Union{Array{T, 1}, Nothing}, rangeθ::Array{T, 2}, 
                                atttraindata::Union{AttributedTrainingData, Nothing},
                                Vhat_set::NamedTuple) where T<:Float64
    # unpack Vhat 
    Vhat = Vhat_set.Vhat
    I_rows = Vhat_set.I_rows
    m = size(Vhat, 2)
    @assert m > k 
    @assert Vhat_set.rangeθ == rangeθ # make sure Vhat is from same setting

    dimθ = size(rangeθ, 1)

    # l_rows = I_rows == nothing ? n : length(I_rows)
    # train an optimal θ value if have training set
    if atttraindata != nothing 
        before = Dates.now()
        @info "Start training θ"
        loss(θ) = loss_fun_reduction(n, k, θ, atttraindata, Vhat; if_deriv = false)[1] 
        loss_deriv(θ) = loss_fun_reduction(n, k, θ, atttraindata, Vhat)[2] 
        function loss_deriv!(G, θ)
            G = loss_deriv(θ)
        end
        @info "Start training"
        # θ_init = rand(Uniform(rangeθ[1, 1], rangeθ[1, 2]), dimθ, 1)
        θ_init = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
        inner_optimizer = LBFGS()
        results = Optim.optimize(loss, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer))
        θ = Optim.minimizer(results)
        @info "Finish training, optimal θ" θ
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        @info "Trained θ, time cost " θ, elapsedmin
    end

    # check if θ is provided or trained
    if θ == nothing && atttraindata == nothing
        θ = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
        @warn "No training info or theta value provided, use random theta within range"
    end

    before = Dates.now()
    # compute H 
    L, _ = laplacian_attributed_L(θ; if_deriv = false)
    # PS: computing H takes 
    H = I_rows == nothing ?  Vhat' * L * Vhat : (Vhat[I_rows, :])' * (L[I_rows, ] * Vhat) 
    @assert size(H) == (m, m) 

    # compute Y, k largest eigenvectors of H
    _, Y = eigs(Symmetric(H); nev=k)

    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    @info "H and Y computation time cost", elapsedmin

    # put Vhat*Y into kmeans
    @info "Start kmeans"
    @time a =  kmeans_reduction(Vhat, Y, k; maxiter = 200)    
    return a, θ
end

function comp_Vhat(n::Int, k::Int, rangeθ::Array{T, 2}; N_sample::Int = 100, precision::T = 0.995, debug::Bool = false) where T<:Float64
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
        L, _ = laplacian_attributed_L(θ; if_deriv = false)
        _, Vk = eigs(L; nev=k)
        Vhat_sample[:, (i-1)*k+1: i*k] = Vk 
    end
    # compute Vhat from truncated SVD from LinearAlgebra
    F = svd(Vhat_sample)
    S = F.S # singular value
    totalsum = sum(S)
    partialsum = [S[1]]
    for i in 2:length(S)
        partialsum = append!(partialsum, partialsum[i-1] + S[i])
        if partialsum[i] > precision * totalsum
            break
        end
    end 
    m = max(length(partialsum), k)
    Vhat =  F.U[:, 1:m] # n by m
    @info "Finish computing Vhat , m = $m"
    return Vhat, S[m+1]
end

function loss_fun_reduction(n::Array{T, 2}, k::Int, θ::Union{Array{T, 1}, T}, atttraindata::AttributedTrainingData, Vhat::Array{T, 2}; 
                            I_rows::Union{Array{Int64,1}, Nothing} = nothing, if_deriv::Bool = true) where T<:Float64
    @info "Evaluate loss func, current θ" θ
    dimθ = length(θ)
    n, m = size(Vhat)
    ntrain = traindata.n
    Apm = traindata.Apm
    # compute Y(θ)
    L, dL = laplacian_attributed_L(θ; if_deriv = if_deriv) # takes 5s
    H = I_rows == nothing ? Vhat' * L * Vhat : (Vhat[I_rows, :])' * (L[I_rows, ] * Vhat) # takes 0.6s, m = 500
    @assert size(H) == (m, m) 
    Λ, Y = eigs(L; nev=k) # takes 3s, k = 40 
    # select training indices 
    Vhat_train_Y = (@view Vhat[1:ntrain, :]) * Y
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
        dloss = Array{Float64, 1}(undef, dimθ)
        for i in 1:dimθ
            Vhat_train_dY = @views Vhat[1:ntrain, :] * dY[:, :, i]
            K = pairwise!(K, SqEuclidean(), Vhat_train_Y, Vhat_train_dY, dims=1)
            dloss[i] = dot(Apm, K)
        end
        dloss = reshape(dloss, dimθ)
    else 
        dloss = nothing
    end
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
    dY = dimθ == 1 ? dropdims(dY; dims = 3) : dY
    return dY
end

function kmeans_reduction(Vhat::Array{T, 2}, Y::Array{T, 2}, k::Int; maxiter::Int = 200) where T<:Float64
    R = kmeans((Vhat*Y)', k; maxiter=maxiter, display=:final)
    @assert nclusters(R) == k # verify the number of clusters
    a = assignments(R) # get the assignments of points to clusters
    return a
end 