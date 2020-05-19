include("./spectral/helpers.jl")
abstract type AbstractTrainingData end
abstract type AbstractTestingData end
abstract type AttributedTrainingData end

struct trainingData <: AbstractTrainingData
    X::Array{Float64, 2} 
    y::Array{Int64, 1}
    n::Int64
    d::Int64 
    Apm::Symmetric{Int64,Array{Int64,2}}
    function trainingData(X::Array{T, 2}, y::Array{Int64, 1}, p::T) where T<:Float64 
        ntrain = Int(floor(n*p))
        Xtrain = X[1:ntrain, :]
        ytrain = y[1:ntrain]
        idtrain = 1:ntrain
        Apm = gen_constraints(Xtrain, ytrain)  #constraint matrix
        return new(Xtrain, ytrain, ntrain, size(Xtrain, 2), Apm)
    end
end

struct testingData <: AbstractTestingData
    X::Array{Float64, 2} 
    y::Array{Int64, 1}
    n::Int64
    d::Int64 
    function testingData(X::Array{Float64, 2}, y::Array{Int64, 1})
        @assert size(X, 1) == length(y)
        return new(X, y, size(X, 1), size(X, 2))
    end
end

struct atttraindata <: AttributedTrainingData
    n::Int # number of training data
    y::Int # training labels
    Apm::Symmetric{Int64,Array{Int64,2}} # constraints matrix
    function atttraindata(n, y, Apm)
        n = 100
        y = ones(n)
        Apm = Symmetric(zeros(n,n))
        return new(n, y, Apm)
    end
end