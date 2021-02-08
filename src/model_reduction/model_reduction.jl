##TODO: approximate version of everyting in src/clustering folder

"""
Compute approximation space for leading eigenvectors (corr. to largest eigenvalues) of a Laplacian matrix
Can select rows on each sampled Vhat(theta) or final Vhat(theta).
Samples theta-values using low-discrepancy sequence. Add spectral gap selection criterion.
Can use DIEM or rank-revealing QR methods to select most important rows of L, for the basis of constructing the Laplacian.

TODO (elouie): need to consider how to select the number of rows for I_rows

Inputs:
  - X: training data (rows)
  - k: the number of target clusters
  - rangeθ: hyperparameter range
Outputs:
  - V_hat: the dimensionally-reduced Vhat
  - I_rows: the selection of rows ordered by importance
"""
function computeApproximationSpace(X, k, rangeθ; pod_first=true)
    # TODO (elouie): Factor out V̂_sample generation
    n, d = size(X)
    dimθ = size(rangeθ, 1)
    @assert size(rangeθ, 2) == 2
    # adjust N_sample if too large
    while (n > 10000) && (k*N_sample > 10000)
        @warn "Too expensive to do svd Vhat_sample of ($n, $(k*N_sample))."
        N_sample = Int(floor(N_sample*0.8))
    end
    # use quasi-random samples
    s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
    N = hcat([next!(s) for i = 1:N_sample]...)' # N_sample * d
    @info "Size of QMC nodes " size(N)
    V̂_sample = Array{Float64, 2}(undef, n, k*N_sample)

    V̂ = computePodReduction(V̂_sample)
    if pod_first
        I_rows = computeRowSelection(V̂)
    else
        I_rows = computeRowSelection(V̂_sample)
    return V̂, I_rows
end

"""
Computes the PCA (POD) reduction of the space.

Inputs:
  - Vhat_sample: training data (rows)
Outputs:
  - V_hat: the dimensionally-reduced version of Vhat_sample
"""
function computePodReduction(Vhat_sample::Array{Float64, 2})
    @info "Computing the dimensional reduction of Vhat."
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
    return Vhat
end

function computeRowSelection(Vhat::Array{Float64, 2})
    @info "Obtaining the DEIM reduction."
    P = qr(u', Val(true)).p
    @info "Obtained row order" P
    return P
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
function computeLoss(X::Array{Float64, 2},  N_sample::Int64, N::Adjoint{Float64,Array{Float64,2}}, Vhat::Array{Float64, 2}; method = "approximate", "exact", I_rows::Array{Int64, 1} = nothing)
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
            Htrue = Vhat' * L.data * Vhat
            H = Vhat[I_rows, :]' * L.data[I_rows, :] * Vhat
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
