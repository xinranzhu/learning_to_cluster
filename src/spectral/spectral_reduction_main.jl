# train an optimal θ
# Xtrain is a small portion of X, with known label ytrain
include("spectral/helpers.jl")

function spectral_reduction_main(X, k, θ, Xtrain = nothing, ytrain = nothing)
    # compute Vhat 
    m = Int(10*k) # could adjust
    Vhat, I_rows = comp_Vhat(X, m, k, rangeθ) #TODO1 below
    n, d = size(X)
    dimθ = length(θ)
    l_rows = length(I_rows)
    # train an optimal θ value if have training set
    if Xtrain != nothing && ytrain != nothing 
        ntrain, dtrain= size(Xtrain)
        ntotal = size(X)
        # generate constraints matrix Apm 
        Apm = gen_constraints(Xtrain, ytrain) 
        # optimize loss fun
        loss(θ) = loss_fun_reduction(θ, Xtrain, Apm, k, Vhat, I_rows...)[1] #TODO2 below
        loss_deriv!(G, θ) = loss_fun_reduction(θ, Xtrain, Apm, k, G, Vhat, I_rows...)[2] #TODO2 below
        θ_init = θ # or rand(dimθ)
        results = optimize((loss, loss_deriv!, θ_init))
        θ = Optim.minimum(results)
    end

    # compute H 
    L = laplacian_L(X, θ, I_rows) #TODO in comp_deriv.jl
    H = (Vhat[I, :])' * (L[I_rows, ] * Vhat)
    @assert size(H) == (m, m) 
    # compute Y, k largest eigenvectors of H
    Y = eigvecs(L)[:, n-k+1:end]

    #TODO3: put VY into kmeans and return clustering results
    # will have to code up our own kmeans, to reuse Vhat - section 3.4, progress report
    R = kmeans_reduction(Vhat, Y, k; maxiter = k)
    return R
end


#TODO 
function comp_Vhat(X, m, k, rangeθ)
    return Vhat, I_rows
end

#TODO
function loss_fun_reduction(θ, Xtrain, Apm, k, Vhat, I_rows...)
    dimθ = length(θ)
    dl = Array{Float64, 1}(undef, dimθ)
    # l::Float64
    return l, dl
end