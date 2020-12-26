using Polynomials
import Polynomials.fit

include("clustering/laplacian.jl")


function fit(x::Array)
end


"""
Wrapper for Clustering.jl kmeans
Inputs:
    - Eigenvectors of L (either exact or inexact)
    - method: defaults to "kmeans"
Outputs:
    - clustering results, e.g. clustering assignments
    -
"""
function cluster(;method = "exact")
    if method == "exact"
        print("in exact")
    elseif method == "approximate"
        print("in approx")
    else
        error("Please choose exact or approximate. Your selection was $method")
    end
end
