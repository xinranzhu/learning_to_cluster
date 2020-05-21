# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain
include("../kernels/kernels.jl")
include("comp_deriv.jl")
include("../datastructs.jl")

# using Optim
using Distances
using TensorOperations
using Dates

"""
k: number of clusters
X:
θ: either initial value of theta
Xtrain:
ytrain: 
"""
function spectral_reduction_main(X::Array{T, 2}, k::Int, θ::Union{Array{T, 1}, T}, rangeθ::Array{T, 2}; 
                                traindata::Union{AbstractTrainingData, Nothing} = nothing, Vhat_set::Union{NamedTuple, Nothing} = nothing) where T<:Float64
    # compute Vhat 
    if Vhat_set == nothing # sample it now
        @info "Start computing Vhat"
        Vhat, I_rows = comp_Vhat(X, k, rangeθ) 
        @info "Vhat size, time cost", size(Vhat)
    else
        Vhat = Vhat_set.Vhat
        I_rows = Vhat_set.I_rows
        @assert Vhat_set.rangeθ == rangeθ # make sure Vhat is from same setting
    end
    n, d = size(X)
    dimθ = length(θ)
    m = size(Vhat, 2)
    @assert m > k 
    # l_rows = I_rows == nothing ? n : length(I_rows)
    # train an optimal θ value if have training set
    if traindata != nothing 
        before = Dates.now()
        @info "Start training θ"
        loss(θ) = loss_fun_reduction(X, k, θ, traindata, Vhat; if_deriv = false)[1] 
        loss_deriv(θ) = loss_fun_reduction(X, k, θ, traindata, Vhat)[2] 
        function loss_deriv!(G, θ)
            G .= loss_deriv(θ)
        end
        @info "Start training"
        if dimθ == 1
            results = Optim.optimize(loss, rangeθ[1, 1], rangeθ[1, 2])
            θ = Optim.minimizer(results)
        else
            # θ_init = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
            θ_init = θ[:]
            # inner_optimizer = GradientDescent(
            #                                     alphaguess = LineSearches.InitialStatic(alpha = 2., scaled = false),
            #                                     linesearch = LineSearches.StrongWolfe())

            nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha = 2., scaled = false),
                                        linesearch=LineSearches.StrongWolfe())
            inner_optimizer = OACCEL(nlprecon=nlprecon, wmax=10)
       
            # inner_optimizer = LBFGS()
            # inner_optimizer = ConjugateGradient()
            results = Optim.optimize(loss, loss_deriv!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer))
            θ = Optim.minimizer(results)
        end
        @info "Finish training, optimal θ" θ
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        @info "Trained θ, time cost " θ, elapsedmin
    end

    if θ == nothing && atttraindata == nothing
        θ = rand(dimθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
        @warn "No training info or theta value provided, use random theta within range"
    end
    
    before = Dates.now()
    # compute H 
    L, dL = laplacian_L(X, θ; I_rows = I_rows, if_deriv = false) 
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
    return a, θ
end

function comp_Vhat(X::Array{T, 2}, k::Int, rangeθ::Array{T, 2}; N_sample::Int = 100, precision::T = 0.995, debug::Bool = false) where T<:Float64
    # before = Dates.now()
    n, d = size(X)
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
    @info "Size of QMC nodes " size(N)
    Vhat_sample = Array{Float64, 2}(undef, n, k*N_sample)
    for i in 1:N_sample
        θ = N[i, :]
        L, dL = laplacian_L(X, θ; if_deriv = false)
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
    m = max(length(partialsum), k)
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
            L = laplacian_L(X, θ; if_deriv=false)[1]
            Htrue = Vhat' * L * Vhat
            H = @views Vhat[I_rows, :]' * L[I_rows, :] * Vhat
            err_cur = norm(Htrue - H)/norm(Htrue)
            err = max(err, err_cur)
        end
        @info "Error from partial rows:" err
    end
    I_rows = nothing
    # after = Dates.now()
    # elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    # Vhat_set = (Vhat = Vhat, range = rangeθ, I_rows = I_rows, N_sample = N_sample, timecost = elapsedmin)
    # save("testVhat.jld", "data", Vhat_set)
    # to check or use later
    # load("testVhat.jld")["data"]
    return Vhat, I_rows
end

# each valuation takes 22s and 50s if deriv.
function loss_fun_reduction(X::Array{T, 2}, k::Int64, θ, traindata::AbstractTrainingData, Vhat::Array{T, 2}; 
                            I_rows::Union{Array{Int64,1}, Nothing} = nothing, if_deriv::Bool = true) where T<:Float64
    dimθ = length(θ)
    n, m = size(Vhat)
    ntrain = traindata.n
    idtrain = 1:ntrain
    Apm = traindata.Apm
    # compute Y(θ)
    L, dL = laplacian_L(X, θ; if_deriv = if_deriv) # 9s
    # H = I_rows == nothing ?  Vhat' * L * Vhat : (Vhat[I_rows, :])' * (L[I_rows, ] * Vhat) # 70s if m = 785
    H = @views Vhat'[:, :] * L[:, :] * Vhat[:, :] # 20s using @views
    @assert size(H) == (m, m) 
    ef = eigen(Symmetric(H), m-k+1:m) # 0.06s, m = 785
    Y = ef.vectors 
    Λ = ef.values
    # select training indices
    Vhat_train_Y = (@view Vhat[1:ntrain, :]) * Y
    # compute loss
    K = Array{Float64, 2}(undef, ntrain, ntrain)
    K = pairwise!(K, SqEuclidean(), Vhat_train_Y, dims=1)
    loss = dot(Apm, K) ./ 2
    if if_deriv
        # compute d(Y(θ))
        if dimθ == 1
            # dH = I_rows == nothing ? Vhat' * dL * Vhat : (Vhat[I_rows, :])' * (dL[I_rows, ] * Vhat)
            # dH = Vhat' * dL * Vhat
            dH = @views Vhat'[:, :] * dL[:, :] * Vhat[:, :]
        else
            dH = Array{Float64, 3}(undef, m, m, dimθ)
            @tensor dH[i,j,k] = Vhat'[i, s] * dL[s, l, k] * Vhat[l, j] 
        end
        dY = comp_dY(Y, Λ, H, dH, dimθ)
        if dimθ > 1
            Vhat_train_dY = Array{Float64, 3}(undef, ntrain, k, dimθ)
            Vhattrain = @view Vhat[idtrain, :]
            @tensor Vhat_train_dY[i, j, k] = Vhattrain[i, s] * dY[s, j, k]
        else
            Vhat_train_dY = (@view Vhat[idtrain, :]) * dY
        end
        K1 = broadcast(-, reshape(Vhat_train_Y, ntrain, 1, k, 1), reshape(Vhat_train_Y, 1, ntrain, k, 1))
        K2 = broadcast(-, reshape(Vhat_train_dY, ntrain, 1, k, dimθ), reshape(Vhat_train_dY, 1, ntrain, k, dimθ))
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