# derivatives w.r.t. θ in R^d
# derivatives of affinity matrix A, Degree matrix, normalized Laplacian and then eigenvectors...

# this changes if paramterization form changes
function affinity_A(X, θ)
    n, d = size(X) 
    dθ = length(θ) 
    if length(θ)>1 # θ = 1/sigma^2
        A = correlation(Gaussian(), θ, X; jitter = 0) 
    else
        A = correlation(Gaussian(), θ[1], X; jitter = 0) 
    end    
    A = Symmetric(A .- Diagonal(diag(A))) # A_ii = 0

    # derivative
    Xdist = broadcast(-, reshape(X, n, 1, d), reshape(X, 1, n, d))
    @assert size(Xdist) == (n, n, d)
    Xdist = - (Xdist .^2) ./ 2
    if dθ == 1
        dA = Array{Float64, 2}(undef, n, n)
        dA = A .* dropdims(sum(Xdist, dims = 3); dims=3)
    elseif dθ == d
        dA = Array{Float64, 3}(undef, n, n, d)
        dA = repeat(A, 1, 1, dθ) .* Xdist
    else
        error("Dimension of parameter should be either 1 or $d")
    end
    return A, dA
end


function degree_D(A, dA)
    n = size(A, 1)
    d = length(θ)
    D = dropdims(sum(A, dims = 2); dims=2)
    dD = dropdims(sum(dA, dims = 2); dims=2)
    return D, dD
end

function laplacian_L(A, dA, D, dD, X, θ)
    sqrtD = 1 ./ sqrt.(D)
    L = Symmetric(Diagonal(sqrtD) * A * Diagonal(sqrtD))

    # dL = 
    

end