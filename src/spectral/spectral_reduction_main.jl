# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain
include("helpers.jl")

using Optim
using Distances
using Dates

"""
k: number of clusters
X:
θ: either initial value of theta
Xtrain:
ytrain: 
"""
function spectral_reduction_main(X, k, θ, Xtrain = nothing, ytrain = nothing)

    # compute Vhat 
    @info "Start computing Vhat"
    before = Dates.now()
    Vhat, I_rows = comp_Vhat(X, k, rangeθ) 
    m = size(Vhat, 2)
    @assert m > k 
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    @info "Vhat size, time cost", size(Vhat), elapsedmin

    n, d = size(X)
    dimθ = length(θ)
    # l_rows = I_rows == nothing ? n : length(I_rows)
    # train an optimal θ value if have training set
    if Xtrain != nothing && ytrain != nothing 
        before = Dates.now()
        ntrain, dtrain= size(Xtrain)
        ntotal = size(X)
        # generate constraints matrix Apm 
        Apm = gen_constraints(Xtrain, ytrain)  #constraint matrix
        # optimize loss fun
        @info "Start training θ"
        loss(θ) = loss_fun_reductionloss_fun_reduction(θ, X, Xtrain, idtrain, Apm, k, Vhat)[1] 
        loss_deriv(θ) = loss_fun_reductionloss_fun_reduction(θ,  X, Xtrain, idtrain, Apm, k, Vhat)[2] 
        function loss_deriv!(G, θ)
            G = loss_deriv(θ)
        end
        θ_init = rand(Uniform(rangeθ[1], rangeθ[2]), dimθ)
        results = optimize((loss, loss_deriv!, θ_init))
        θ = Optim.minimum(results)
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        @info "Trained θ, time cost " θ, elapsedmin
    end

    before = Dates.now()
    # compute H 
    L, _ = laplacian_L(X, θ, I_rows) 
    H = I_rows == nothing ?  Vhat' * L * Vhat : (Vhat[I_rows, :])' * (L[I_rows, ] * Vhat)
    @assert size(H) == (m, m) 

    # compute Y, k largest eigenvectors of H
    ef = eigen(Symmetric(H), m-k+1:m)
    Y = ef.vectors
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    @info "H and Y computation time cost", elapsedmin

    # put Vhat*Y into kmeans
    @info "Start kmeans"
    @time a =  kmeans_reduction(Vhat, Y, k; maxiter = 200)    
    return a
end

function comp_Vhat(X, k, rangeθ; N_sample = 100, precision = 0.995, debug = false, num_rows = nothing)
    n, d = size(X)
    dimθ = size(rangeθ, 1)
    @assert size(rangeθ, 2) == 2 
    # use quasi-random samples
    s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
    N = hcat([next!(s) for i = 1:N_sample]...)' # N_sample * d
    Vhat_sample = Array{Float64, 2}(undef, n, k*N_sample)
    for i in 1:N_sample
        θ = N[i, :]
        L, _ = laplacian_L(X, θ)
        ef = eigen(Symmetric(L), n-k+1:n)
        Vhat_sample[:, (i-1)*k+1: i*k] = ef.vectors
    end
    # compute Vhat from truncated SVD 
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
    m = length(partialsum)
    Vhat =  F.U[:, 1:m] # n by m

    # select important rows from Vhat_sample

    # F = pqrfact(Vhat_sample') # F[:k] gives the rank
    # num_rows = num_rows == nothing ? F[:k] : num_rows
    # I_rows = findall(x->x<num_rows, F.p)
    if debug == true
        global err
        err = 0.
        for i in 1:N_sample
            θ = N[i, :]
            L, _= laplacian_L(X, θ)
            Htrue = Vhat' * L * Vhat
            H = @views Vhat[I_rows, :]' * L[I_rows, :] * Vhat
            err_cur = norm(Htrue - H)/norm(Htrue)
            err = max(err, err_cur)
        end
        @info "Error from partial rows:" err
    end
    I_rows = nothing
    return Vhat, I_rows
end

function loss_fun_reduction(θ, X, Xtrain, idtrain, Apm, k, Vhat; I_rows = nothing)
    dimθ = length(θ)
    ntrain, d = size(Xtrain)
    n, m = size(Vhat)
    # compute Y(θ)
    L, dL = laplacian_L(X, θ) 
    H = I_rows == nothing ?  Vhat' * L * Vhat : (Vhat[I_rows, :])' * (L[I_rows, ] * Vhat)
    @assert size(H) == (m, m) 
    ef = eigen(Symmetric(H), m-k+1:m)
    Y = ef.vectors 
    Λ = ef.values
    # select training indices
    Vhat_train_Y = Vhat[idtrain, :] * Y
    # compute loss
    K = Array{Float64, 2}(undef, ntrain, ntrain)
    K = pairwise!(K, SqEuclidean(), Vhat_train_Y, dims=1)
    loss = dot(Apm, K) ./ 2

    # compute d(Y(θ))
    if dimθ == 1
        # dH = I_rows == nothing ? Vhat' * dL * Vhat : (Vhat[I_rows, :])' * (dL[I_rows, ] * Vhat)
        dH = Vhat' * dL * Vhat
    else
        dH = Array{Float64, 3}(undef, m, m, dimθ)
        @tensor dH[i,j,k] = Vhat'[i, s] * dL[s, l, k] * Vhat[l, j] 
    end
    dY = comp_dY(Y, Λ, H, dH, dimθ)
    if dimθ == 1
        Vhat_train_Y = Vhat[idtrain, :] * Y
        Vhat_train_dY = Vhat[idtrain, :] * dY
        # K = Array{Float64, 2}(undef, ntrain, ntrain)
        K = pairwise!(K, SqEuclidean(), Vhat_train_Y, Vhat_train_dY, dims=1)
        dloss = dot(Apm, K)
    else
        dloss = Array{Float64, 1}(undef, dimθ)
        Vhat_train_Y1 = @views Vhat[idtrain, :] * Y
        for i in 1:dimθ
            Vhat_train_dY = @views Vhat[idtrain, :] * dY[:, :, i]
            K = pairwise!(K, SqEuclidean(), Vhat_train_Y, Vhat_train_dY, dims=1)
            dloss[i] = dot(Apm, K)
        end
    end
    return loss, dloss
end

function comp_dY(Y, Λ, H, dH, dimθ)
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

function kmeans_reduction(Vhat, Y, k; maxiter = 200)    
    R = kmeans((Vhat*Y)', k; maxiter=maxiter, display=:iter)
    @assert nclusters(R) == k # verify the number of clusters
    a = assignments(R) # get the assignments of points to clusters
    return a
end 