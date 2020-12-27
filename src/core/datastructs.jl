using LinearAlgebra

abstract type AbstractTrainingData end
abstract type AbstractTestingData end
abstract type AttributedTrainingData end

struct trainingData <: AbstractTrainingData
    X::Array{Float64, 2}
    y::Array{Int64, 1}
    n::Int64
    d::Int64
    LinkConstraintsMatrix::Symmetric{Int64,Array{Int64,2}}
    function trainingData(X::Array{T, 2}, y::Array{Int64, 1}, ntrain::Int64; C::Int64 = 1) where T<:Float64
        Xtrain = X[1:ntrain, :]
        ytrain = y[1:ntrain]
        idtrain = 1:ntrain
        LinkConstraintsMatrix = gen_constraints(ntrain, ytrain; C = C)  #constraint matrix
        return new(Xtrain, ytrain, ntrain, size(Xtrain, 2), LinkConstraintsMatrix)
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

struct attributedTrainingData <: AttributedTrainingData
    n::Int # number of training data
    id::Array{Int64, 1}
    y::Array{Int64, 1} # training labels
    LinkConstraintsMatrix::Symmetric{Int64,Array{Int64,2}} # constraints matrix
    function attributedTrainingData(n::Int64, id::Array{Int64, 1}, y::Array{Int64, 1}; C::Int64 = 1)
        LinkConstraintsMatrix = gen_constraints(n, y; C = C)
        return new(n, id, y, LinkConstraintsMatrix)
    end
end

"""
"""
function gen_constraints(n::Int64, y::Array{Int64, 1}; C::Int64 = 1)
    LinkConstraintsMatrix = Array{Float64, 2}(undef, n, n)
    R = CartesianIndices(LinkConstraintsMatrix)
    for I in R
        i, j = Tuple(I)
        if i == j
            constraint = 0
        elseif y[i] == y[j]
            constraint = C
        else
            constraint = -1
        end
        LinkConstraintsMatrix[I] = constraint
    end
    return Symmetric(LinkConstraintsMatrix)
end
