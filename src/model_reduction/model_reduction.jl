##TODO: approximate version of everyting in src/clustering folder

"""
    computeApproximationSpace(X::Array{T, 2}, k::Integer, rangeθ::Array{T, 2}[, pod_first::Bool=true, N_sample::Integer=100])

Compute approximation space for leading eigenvectors (corr. to largest eigenvalues) of a Laplacian matrix. Can select
rows on each sampled Vhat(theta) or final Vhat(theta). Samples theta-values using low-discrepancy sequence. Add
spectral gap selection criterion. Can use DIEM or rank-revealing QR methods to select most important rows of L, for
the basis of constructing the Laplacian.

# Arguments:
- `X::Array{T, 2}`: training data (rows).
- `k::Integer`: the number of target clusters.
- `rangeθ::Array{T, 2}`: hyperparameter range.
- `pod_first::Bool`: whether to perform POD before DEIM reduction.

# Returns:
- `V̂::Array{T, 2}`: the dimensionally-reduced V̂.
- `I_rows::Array{Integer, 1}`: the selection of rows ordered by importance.
"""
function computeApproximationSpace(X::Array{T, 2}, k::Integer, rangeθ::Array{T, 2}; pod_first::Bool=true, N_sample::Integer=100) where T <: Float64
    # TODO (elouie): Factor out V̂_sample generation
    V̂_sample = computeSamples(X, k, rangeθ)

    # Compute the POD-reduction prior to DEIM.
    V̂ = computePodReduction(V̂_sample, k)

    # TODO (elouie): need to consider how to select the number of rows for I_rows
    if pod_first
        I_rows = computeRowSelection(V̂)
    else
        I_rows = computeRowSelection(V̂_sample)
    return V̂, I_rows
end

"""
    computeSamples(X::Array{T, 2}, k::Integer, rangeθ::Array{T, 2}[, N_sample::Integer=100])

Computes a sample set of points for the provided θ parameters.

Note: computations are limited to fewer than 10,000 samples for performance.

# Arguments
- `X::Array{T, 2}`: training data (rows).
- `k::Integer`: the number of target clusters.
- `rangeθ::Array{T, 2}`: hyperparameter range.
- `pod_first::Bool`: whether to perform POD before DEIM reduction.

# Returns:
- `V̂_sample::Array{T, 2}`: matrix of (n × k⋅N_samples) samples of V̂.
"""
function computeSamples(X::Array{T, 2}, k::Integer, rangeθ::Array{T, 2}; N_sample::Integer=100)
    n, d = size(X)
    dimθ = size(rangeθ, 1)
    @assert size(rangeθ, 2) == 2
    # adjust N_sample if too large
    while (n > 10000) && (k*N_sample > 10000)
        @warn "Cannot perform POD reduction on V̂_sample size of ($n, $(k*N_sample))."
        N_sample = Int(floor(N_sample*0.8))
    end
    # use quasi-random samples
    s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
    N = hcat([next!(s) for i = 1:N_sample]...)' # N_sample * d
    @info "Size of QMC nodes " size(N)
    V̂_sample = Array{Float64, 2}(undef, n, k*N_sample)
    return V̂_sample
end

"""
    computePodReduction(V̂_sample::Array{Float64, 2}, k::Integer)

Computes the PCA (POD) reduction of the space. Keeps up to `k` eigenvalues that total 99.95%

# Arguments:
- `V̂_sample::Array{Float64, 2}`: training data (rows).
- `k::Integer`: maximum dimension of POD-reduction.

# Returns:
- V̂: V̂_sample dimensionally-reduced.
"""
function computePodReduction(V̂_sample::Array{Float64, 2}, k::Integer)
    @info "Computing the dimensional reduction of Vhat."
    F = svd(V̂_sample)
    S = F.S # singular value
    precision_sum = precision * sum(S)
    partialsum = [S[1]]
    for i in 2:length(S)
        partialsum = append!(partialsum, partialsum[i-1] + S[i])
        if partialsum[i] > precision_sum
            break
        end
    end
    m = max(length(partialsum), k)
    V̂ =  F.U[:, 1:m] # n by m
    return V̂
end

"""
    computeRowSelection(V̂::Array{Float64, 2})

Computes the QR-DEIM reduction using QR-pivoting.

# Arguments:
- `V̂_sample::Array{Float64, 2}`: training data (rows).
- `k::Integer`: maximum dimension of POD-reduction.

# Returns:
- V̂: V̂_sample dimensionally-reduced.
"""
function computeRowSelection(V̂::Array{Float64, 2})
    @info "Obtaining the DEIM reduction."
    # Uses Householder under the hood. pqrfact() from LowRankApproximation uses Householder / Rank Revealing.
    P = qr(u', Val(true)).p
    @info "Obtained row order" P
    return P
end

"""
    computeLoss(X::Array{Float64, 2},  N_sample::Int64, N::Adjoint{Float64,Array{Float64,2}}, V̂::Array{Float64, 2}; I_rows::Array{Int64, 1} = nothing)

Computes the normalized Rayleigh-Ritz spectral clustering loss between the reduced and original space.

Arguments:
- `X::Array{Float64, 2}`: original dataset to cluster.
- `N_sample`: number of samples to generate for approximate subspace.
- `V̂::Array{Float64, 2}`: approximate space.
- `I_rows`: if supplied, uses the rows selected for model reduction.

Returns:
- `loss::Array{Float64, 1}`: An array of the loss per parameter.
"""
function computeLoss(X::Array{Float64, 2},  N_sample::Int64, N::Adjoint{Float64,Array{Float64,2}}, V̂::Array{Float64, 2}; I_rows::Array{Int64, 1} = nothing)
    if I_rows == nothing # Exact loss
        # use lossfun from clustering/lossfun.jl
    else
        @info "Calculating the DEIM model reduction error..."
        loss = Array{Float64, 1}(undef, N_sample)
        @showprogress for i in 1:N_sample
            θ = N[i, :]
            L = laplacian_L(X, θ; I_rows = nothing, derivative=false)[1]
            # L is type Symmetric, which is great for storage, but makes a sparse
            # type that _does not_ use Julia block multiply optimizations. L.data
            # returns the original matrix, which results in a very fast computation.
            Htrue = Vhat' * L.data * V̂
            H = V̂[I_rows, :]' * L.data[I_rows, :] * V̂
            loss[i] = norm(Htrue - H)/norm(Htrue)
        end
        @info "Max DEIM loss: " maximum(loss)
        @info "Average DEIM loss: " mean(loss)
        @info "Median DEIM loss: " median(loss)
        @info "Minimum DEIM loss: " minimum(loss)
        return loss
    end
end

"""
Note: exact lossfun has already been implemented in clustering/lossfun.jl
Inputs:
    - X: entire dataset, what to cluster
    - trainingData: subset of data with ground truth labels
    - method: "approximate" or "exact"
Outputs:
    - loss, dloss
"""
function computeSampleApproximationLoss()
    # TODO (elouie): Review whether there is a faster method for getting H_true
    H_true = V̂' * L.data * V̂
    H = Vhat[I_rows, :]' * L.data[I_rows, :] * Vhat
    loss[i] = norm(Htrue - H)/norm(Htrue)
end

"""
Optimize Laplacian hyperaparmeters using Optim.optimize

Inputs:
     - loss and derivative functions
"""
function optimizeLaplacian()
end

"""
Computes Y and H for forming approximate eigenspace for L
Inputs:
  - L: actual Laplacian matrix
  - V_hat: approximation space for eigenvectors
  - I_rows: subset of rows/indices (section 3.2 in report)
Outputs:
  - Y: weights for combining columns of V_hat

"""
function computeApproximateEigenvectors()
end
