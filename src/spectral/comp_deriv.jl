# derivatives w.r.t. θ in R^d
# derivatives of affinity matrix A, Degree matrix, normalized Laplacian and then eigenvectors...

# this changes if paramterization form changes
# turns out we always need full A, even in L[I_rows, :] since we need full D there
function affinity_A(X, θ; if_deriv = true)
    # @info "enter affinity A"
    n, d = size(X) 
    dimθ = length(θ)
    # if I_rows == nothing 
    # X1 = X;
    n1 = n
    if length(θ)>1 # θ = 1/sigma^2
        A = correlation(Gaussian(), θ, X) 
    else
        A = correlation(Gaussian(), θ[1], X) 
    end    
    A = Symmetric(A .- Diagonal(diag(A))) # A_ii = 0
    # else
    #     X1 = X[I_rows, :]; n1 = length(I_rows)
    #     if length(θ)>1 # θ = 1/sigma^2
    #         A = cross_correlation(Gaussian(), θ, X1, X) 
    #     else
    #         A = cross_correlation(Gaussian(), θ[1], X1, X) 
    #     end  
    #     A[I_rows, I_rows] .-=  Diagonal(diag(A[I_rows, I_rows]))
    #     @assert size(A) == (n1, n) 
    # end
    # derivative
    # @info "finish computing A"
    if if_deriv
        Xdist = broadcast(-, reshape(X, n1, 1, d), reshape(X, 1, n, d))
        @assert size(Xdist) == (n1, n, d)
        Xdist = - (Xdist .^2) ./ 2
        if dimθ == 1
            dA = Array{Float64, 2}(undef, n1, n)
            dA = A .* dropdims(sum(Xdist, dims = 3); dims=3)
        elseif dimθ == d
            dA = Array{Float64, 3}(undef, n1, n, d)
            # dA = repeat(A, 1, 1, dimθ) .* Xdist
            dA = broadcast(*, reshape(A, n1, n, 1), Xdist)
        else
            error("Dimension of parameter should be either 1 or $d")
        end
    else 
        dA = nothing
    end
    return A, dA 
end

# don't have to change for different paramterization form
function degree_D(X, θ; if_deriv = true)
    # @info "enter degree D"
    A, dA = affinity_A(X, θ; if_deriv = if_deriv)
    # @info "finish computing A"  
    dimθ = length(θ)
    D = dropdims(sum(A, dims = 2); dims=2)
    # @info "finish computing D"
    dD = if_deriv ? dropdims(sum(dA, dims = 2); dims=2) : nothing
    # @info "finish computing dD"
    return D, dD, A, dA
end

function laplacian_L(X, θ; I_rows = nothing, if_deriv = true)
    # @info "enter laplacian"
    D, dD, A, dA = degree_D(X, θ; if_deriv = if_deriv)
    # @info "finish computing D, A" size(D), size(dD), size(A), size(dA)
    sqrtD = 1 ./ sqrt.(D)
    n = length(D)
    d = length(θ)
    if I_rows == nothing
        n1 = n
        sqrtDleft = sqrtD
        L = Symmetric(Diagonal(sqrtD) * A * Diagonal(sqrtD))
    else 
        n1 = length(I_rows)
        A = A[I_rows, :]
        sqrtDleft = sqrtD[I_rows]
        L = Diagonal(sqrtDleft) * A * Diagonal(sqrtD)
    end
    # @info "finish computing L" size(L)
    if if_deriv
        invDdD = broadcast(*, reshape(1 ./ D, n, 1), dD)
        dA = I_rows == nothing ? dA : reshape(dA, n, n, d)[I_rows, :, :]
        invDdDleft = I_rows == nothing ? invDdD : invDdD[I_rows, :]
        dL = broadcast(*, reshape(sqrtDleft, n1, 1), dA) 
        dL = reshape(broadcast(*, reshape(sqrtD, 1, n), dL), n1, n, d)
        dL .-= broadcast(*, reshape(invDdDleft, n1, 1, d), L)./2 .+ broadcast(*, reshape(invDdD, 1, n, d), L)./2
        if d == 1 
            dL = dropdims(dL; dims = 3)
        end
    else
        dL = nothing
    end
    return L, dL
end

# function laplacian_L_attributed()
