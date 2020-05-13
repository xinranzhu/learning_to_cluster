# derivatives w.r.t. θ in R^d
# derivatives of affinity matrix A, Degree matrix, normalized Laplacian and then eigenvectors...

# this changes if paramterization form changes
function affinity_A(X, θ; I_rows = nothing)
    n, d = size(X) 
    dimθ = length(θ)
    if I_rows == nothing 
        X1 = X; n1 = n
        if length(θ)>1 # θ = 1/sigma^2
            A = correlation(Gaussian(), θ, X) 
        else
            A = correlation(Gaussian(), θ[1], X) 
        end    
        A = Symmetric(A .- Diagonal(diag(A))) # A_ii = 0
    else
        X1 = X[I_rows, :]; n1 = length(I_rows)
        if length(θ)>1 # θ = 1/sigma^2
            A = cross_correlation(Gaussian(), θ, X1, X) 
        else
            A = cross_correlation(Gaussian(), θ[1], X1, X) 
        end  
        A[I_rows, I_rows] .-=  Diagonal(diag(A[I_rows, I_rows]))
        @assert size(A) == (n1, n) 
    end
    # derivative
    Xdist = broadcast(-, reshape(X1, n1, 1, d), reshape(X, 1, n, d))
    @assert size(Xdist) == (n1, n, d)
    Xdist = - (Xdist .^2) ./ 2
    if dimθ == 1
        dA = Array{Float64, 2}(undef, n1, n)
        dA = A .* dropdims(sum(Xdist, dims = 3); dims=3)
    elseif dimθ == d
        dA = Array{Float64, 3}(undef, n1, n, d)
        dA = repeat(A, 1, 1, dimθ) .* Xdist
    else
        error("Dimension of parameter should be either 1 or $d")
    end
    return A, dA
end


# don't have to change for different paramterization form
function degree_D(X, θ; I_rows = nothing)
    A, dA = affinity_A(X, θ)
    if I_rows == nothing
        AI = nothing
        dAI = nothing
    else
        AI = A[I_rows, :]
        dAI = dA[I_rows, :, :]
    end
    dimθ = length(θ)
    D = dropdims(sum(A, dims = 2); dims=2)
    dD = dropdims(sum(dA, dims = 2); dims=2)
    return D, dD, A, dA, AI, dAI
end

#TODO
function laplacian_L(X, θ; I_rows = nothing)
    D, dD, A, dA, AI, dAI = degree_D(X, θ; I_rows = I_rows)
    sqrtD = 1 ./ sqrt.(D)

    L = 
    # L = Symmetric(Diagonal(sqrtD) * A * Diagonal(sqrtD))

    # dL = not sure about the fastest way 

end
