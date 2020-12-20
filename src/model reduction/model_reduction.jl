"""
Compute approximation space for leading eigenvectors (corr. to largest eigenvalues) of a Laplacian matrix
Can select rows on each sampled Vhat(theta) or final Vhat(theta).
Samples theta-values using low-discrepancy sequence. Add spectral gap selection criterion.
Can use DIEM or rank-revealing QR methods to select most important rows of L, for the basis of constructing the Laplacian.

Inputs: Laplacian, sampling_args
  - X: training data (rows)
  - k: target clusters
  - rangeθ: hyperparameter range
Outputs: V_hat, I_rows
"""
function computeApproximationSpace()
end

"""
Inputs:
    - X: entire dataset, what to cluster
    - trainingData: subset of data with ground truth labels
    - method: "approximate" or "exact"
Outputs:
    - loss, dloss
"""
function computeLoss(; method = "approximate", "exact")
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
