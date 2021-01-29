##TODO: approximate version of everyting in src/clustering folder

"""
Compute approximation space for leading eigenvectors (corr. to largest eigenvalues) of a Laplacian matrix
Can select rows on each sampled Vhat(theta) or final Vhat(theta).
Samples theta-values using low-discrepancy sequence. Add spectral gap selection criterion.
Can use DIEM or rank-revealing QR methods to select most important rows of L, for the basis of constructing the Laplacian.

Inputs: training data, sampling_args
  - X: training data (rows)
  - k: target clusters
  - rangeθ: hyperparameter range
Outputs: V_hat, I_rows
"""
function computeApproximationSpace()
end

"""
Note: exact lossfun has already been implemented in clustering/lossfun.jl
Inputs:
    - X: entire dataset, what to cluster
    - trainingData: subset of data with ground truth labels
    - method: "approximate" or "exact"
    - V_hat
Outputs:
    - loss, dloss
"""
function computeLoss(; method = "approximate", "exact")
    if method == "approximate"
        #TODO
        # L = laplacian_L(X, θ; I_rows = nothing, derivative = true)
    elseif method == "exact"
        # use lossfun from clustering/lossfun.jl
    else
        error("method not defined")
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
