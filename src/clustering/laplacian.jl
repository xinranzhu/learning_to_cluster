include("../kernels/kernels.jl")
using TensorOperations

# Laplacian and relevant matrix computations (including derivatives)

# derivatives w.r.t. θ in R^d
# derivatives of affinity matrix A, Degree matrix, normalized Laplacian and then eigenvectors...

# this changes if paramterization form changes
# turns out we always need full A, even in L[I_rows, :] since we need full D there
function affinity_A(X, θ; derivative = true)
    # @info "enter affinity A"
    n, d = size(X)
    dimθ = length(θ)
    n1 = n
    A = KernelMatrix(RBFKernelModule(θ, 1.0, 0.0), X)
    A = Symmetric(A .- Diagonal(diag(A))) # A_ii = 0
    if derivative
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
function degree_D(X, θ; derivative = true)
    # @info "enter degree D"
    A, dA = affinity_A(X, θ; derivative = derivative)
    # @info "finish computing A"
    dimθ = length(θ)
    D = dropdims(sum(A, dims = 2); dims=2)
    # @info "finish computing D"
    dD = derivative ? dropdims(sum(dA, dims = 2); dims=2) : nothing
    # @info "finish computing dD"
    return D, dD, A, dA
end

"""
D^-1/2 * A * D^-1/2
"""
function laplacian_L(X, θ; I_rows = nothing, derivative = true)
    # @info "enter laplacian"
    D, dD, A, dA = degree_D(X, θ; derivative = derivative)
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
    if derivative
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

"""
Compute derivative of eigenvectors of Laplacian w.r.t hyperparameters.
If normalized flag is set to true, then computes entry-wise derivative of normalized V.
"""
function comp_dV(V::Array{T, 2}, Λ::Array{T, 1}, L::Symmetric{T,Array{T,2}},
    dL::Union{Array{T, 2}, Array{T, 3}}, dimθ::Int64; normalized = true) where T<:Float64
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
    if normalized
        row_norms = mapslices(norm, V; dims=2)
        row_norms_squared = row_norms .^ 2
        dV_normalized = Array{Float64, 3}(undef, n, k, dimθ)
        for i = 1:n
            for j = 1:k
                total = zeros(dimθ)
                for w = [1:j-1; j+1:k]
                    total += V[i, w] * (dV[i, j, :]*V[i, w] - V[i, j]*dV[i, w, :])
                end
                dV_normalized[i, j, :] = total ./ row_norms_squared[i]^(3/2)
            end
        end
        dV = dV_normalized
    end
    return dV
end
