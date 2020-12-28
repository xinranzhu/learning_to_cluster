using Polynomials
import Polynomials.fit

include("core/datastructs.jl")
include("clustering/laplacian.jl")
include("clustering/lossfun.jl")
include("clustering/spectral.jl")
include("clustering/kmeans_match_labels.jl")

"""
Evaluate and compare variants of SC, including normalized and unnormalized,
trained and untrained, in addition to k-means.
INPUTS:
    - train: whether or not to train for θ using labels, if false, then
            θ_init defaults to ones(d)
    - ntrials: number of trials
"""
function evaluate_spectral_clustering(data, label; frac_train = 0.3, train = false, ntrials = 20, time_limit = 100)
    n = size(data, 1)
    k = length(unique(label))
    @assert minimum(label)==1
    d = size(data, 2)
    @assert frac_train < 1; @assert frac_train > 0
    ntrain = floor(n * frac_train)
    mytrain = trainingData(data, label, Int(ntrain))

    result = Dict()
    # untrained normalized and unnormalized SC
    θ_init = ones(d)
    function SC(theta; normalized = false)
        L = Matrix(laplacian_L(data, theta)[1])
        sc_assignment = cluster_spectral(L, k; normalized = normalized)
        sc_max_acc, matched_assignment = bipartite_match_labels(sc_assignment, label, 6)
        return sc_max_acc
    end
    if train == false
        result[:U] = SC(θ_init, normalized = false)
        result[:N] = SC(θ_init, normalized = true)
    elseif train == true
        # Semisupervised/trained normalized and unnormalized SC
        rangeθ = hcat(0.01*reshape(ones(d), d, 1), 100*reshape(ones(d), d, 1))
        inner_optimizer = LBFGS()
        #Unnormalized and trained SC
        q(θ) = loss_fun(data, k, d, θ, mytrain; normalized = false)[1]
        dq(θ) = loss_fun(data, k, d, θ, mytrain; normalized = false)[2]
        function dq!(G, θ)
            G .= dq(θ)
        end
        opt_U = Optim.optimize(q, dq!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer), Optim.Options(show_trace=true, time_limit = time_limit))
        optθ_U = Optim.minimizer(opt_U)
        @info optθ_U
        total_U = 0
        for i = 1:ntrials
            total_U += SC(optθ_U, normalized = false)
        end
        result[:U] = total_U/ntrials
        # Normalized and trained SC
        h(θ) = loss_fun(data, k, d, θ, mytrain; normalized = true)[1]
        dh(θ) = loss_fun(data, k, d, θ, mytrain; normalized = true)[2]
        function dh!(G, θ)
            G .= dh(θ)
        end
        opt_N = Optim.optimize(h, dh!, rangeθ[:,1], rangeθ[:,2], θ_init, Fminbox(inner_optimizer), Optim.Options(show_trace=true, time_limit = time_limit))
        optθ_N = Optim.minimizer(opt_N)
        @info optθ_N
        total_N = 0
        for i = 1:ntrials
            total_N += SC(optθ_N, normalized = true)
        end
        result[:N] = total_N/ntrials
    else
        error("train flag must be true or false")
    end
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
