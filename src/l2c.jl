using Polynomials
import Polynomials.fit

include("core/datastructs.jl")
include("clustering/laplacian.jl")
include("clustering/lossfun.jl")
include("clustering/spectral.jl")
include("clustering/kmeans_match_labels.jl")
include("utils/preprocess.jl")


"""
Evaluate and compare variants of SC, including normalized and unnormalized,
trained and untrained, in addition to k-means.
INPUTS:
    - train: whether or not to train for θ using labels, if false, then
            θ_init defaults to ones(d)
    - ntrials: number of trials
"""
function evaluate_spectral_clustering(data, label; frac_train = 0.3, train::Bool = false, normalized::Bool = true, ntrials = 20, time_limit = 100)
    n = length(label)
    k = length(unique(label))
    @assert minimum(label)==1
    d = size(data, 2)
    @assert frac_train < 1; @assert frac_train > 0
    ntrain = floor(n * frac_train)
    mytrain = trainingData(data, label, Int(ntrain))

    result = Dict()
    # untrained normalized and unnormalized SC
    θ_init = ones(d)
    function SC(theta; sc_normalized = false)
        L = Matrix(laplacian_L(data, theta)[1])
        sc_assignment = cluster_spectral(L, k; normalized = sc_normalized)
        sc_max_acc, matched_assignment = bipartite_match_labels(sc_assignment, label, 6)
        return sc_max_acc
    end
    total = 0.0
    if train == false
        for i = 1:ntrials
            total += SC(θ_init, sc_normalized = normalized)
        end
    else
        # Semisupervised/trained normalized and unnormalized SC
        rangeθ = hcat(0.01*reshape(ones(d), d, 1), 100*reshape(ones(d), d, 1))
        inner_optimizer = LBFGS()
        #Unnormalized and trained SC
        #q(θ) = loss_fun(data, k, d, θ, mytrain; normalized = normalized)[1]
        #dq(θ) = loss_fun(data, k, d, θ, mytrain; normalized = normalized)[2]
        q(θ) = - loss_fun_eigengap(data, k, d, θ)[1]
        dq(θ) = - loss_fun_eigengap(data, k, d, θ)[2][:]
        function dq!(G, θ)
            G .= dq(θ)
        end
        opt = Optim.optimize(q, dq!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer), Optim.Options(show_trace=true, time_limit = time_limit))
        optθ = Optim.minimizer(opt)
        @info optθ
        for i = 1:ntrials
            total += SC(optθ, sc_normalized = normalized)
        end
    end
    result[:Ans] = total / ntrials
    return result
end

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
