using Distances
using LinearAlgebra
using SparseArrays
import Base: size

@doc raw"""
    AbstractCorrelation

An abstract type for all correlation functions.
"""
abstract type AbstractCorrelation end

(k::AbstractCorrelation)(x::Array, y::Array, θ) = k(distance(k, θ)(x, y),  θ)

"""
Computes matrix of pairwise distances between x and y, which contains points
arranged row-wise
"""
function distancematrix(k::AbstractCorrelation, θ, x, y=x; dims=1)
    ret = Array{Float64}(undef, size(x, dims), size(x, dims))
    computeDists!(ret, k, θ, x, y; dims=dims)
    return ret
end

"""
x, y: arrays of points in R^d
θ: 1D array
"""
function computeDists!(out, k::AbstractCorrelation, θ, x, y=x; dims=1)
    if typeof(θ)<:Array
        @assert max(size(θ, 1), size(θ, 2)) == size(x, 2) == size(y, 2)
    end
    dist = distance(k, θ)
    try
        out .= pairwise!(out, dist, x, y, dims = dims)
    catch(MethodError)
        out .= pairwise!(out, dist, reshape(x,size(x, 1), size(x, 2)), reshape(y, size(y, 1), size(y, 2)), dims = dims)
    end
        return nothing
end

"""
jitter: added to diagonal of kernel matrix to ensure positive-definiteness
dims: 1 if data points are arranged row-wise and 2 if col-wise
"""
function correlation(k::AbstractCorrelation, θ, x; jitter = 0, dims=1)
    ret = Array{Float64}(undef, size(x, dims), size(x, dims))
    correlation!(ret, k, θ, x, jitter = jitter)
    # mycorrelation!(ret, k, θ, x, jitter = jitter)
    return ret
end


function cross_correlation(k::AbstractCorrelation, θ, x, y; dims=1)
    ret = Array{Float64}(undef, size(x, dims), size(y, dims))
    cross_correlation!(ret, k, θ, x, y;dims=dims)
    return ret
end

function cross_correlation!(out, k::AbstractCorrelation, θ, x, y; dims = 1)
    x = reshape(x, size(x, 1), size(x, 2)); y = reshape(y, size(y, 1), size(y, 2)) #2D arrays
    dist = distance(k, θ)
    pairwise!(out, dist, x, y, dims=dims)
    out .= (τ -> k(τ, θ)).(out)
    return nothing
end

function correlation!(out, k::AbstractCorrelation, θ, x; jitter = 0, dims=1)
    x = reshape(x, size(x, 1), size(x, 2))
    # @info "Current θ and type " θ, typeof(θ)
    dist = distance(k, θ)
    pairwise!(out, dist, x, dims=dims)
    out .= (τ -> k(τ, θ)).(out)
    if jitter != 0
        out[diagind(out)] .+= jitter
        # out ./= out[1, 1] #covariance must be in [0, 1]
    end
    return nothing
end



function mycorrelation!(out, k::AbstractCorrelation, θ, x; jitter = 0, dims=1)
    x = reshape(x, size(x, 1), size(x, 2))
    n, d = size(x)
    @info "Current θ and type " θ, typeof(θ)
    Wsq(x, y, θ) = sum((x - y).^2 .* θ)
    wq_fixed(x, y) =  exp(- Wsq(x, y, θ) / 2)
    for j = 1:n
        bj = view(x, j, :)
        for i = 1:n
            ai = view(x, i, :)
            out[i, j] = wq_fixed(ai, bj)
        end
    end
    return nothing
end

@doc raw"""
"""
struct FixedParam{K<:AbstractCorrelation,T} <: AbstractCorrelation
    k::K
    θ::T
end
FixedParam(k, θ...) = FixedParam(k, θ)

(k::FixedParam)(τ) = k.k(τ, k.θ...)
distance(k::FixedParam) = distance(k.k, k.theta...)

@doc raw"""
    Gaussian
"""
struct Gaussian <: AbstractCorrelation end
const RBF = Gaussian
const SqExponential = RBF

#WARNING: these two functions are purely meant to be used in conjunction with (k::AbstractCorrelation)(x::Array, y::Array, θ)
#These are not standalone functions!
(::Gaussian)(τ::Real, θ::Real) = exp(- τ * θ / 2)
(::Gaussian)(τ::Real, ::AbstractVector) = exp(- τ / 2) #theta already taken into account in computation of tau
(::Gaussian)(τ::Real, ::Any) = exp(- τ / 2) #theta already taken into account in computation of tau

distance(::Gaussian, θ::Real) = SqEuclidean()
# distance(::Gaussian, θ::AbstractVector) = WeightedSqEuclidean(θ)
distance(::Gaussian, θ::AbstractVector) = WeightedSqEuclidean(θ)
distance(::Gaussian, θ) = WeightedSqEuclidean(θ)
