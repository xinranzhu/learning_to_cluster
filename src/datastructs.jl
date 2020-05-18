struct trainingData <: TrainingData
    X::Array{Float64, 2} 
    y::Array{Float64, 1}
    n::Int64
    d::Int64 
    Apm::Array{Int64, 2} 
    function trainingData(X, y, p) 
        ntrain = Int(floor(n*p))
        Xtrain = X[1:ntrain, :]
        ytrain = y[1:ntrain]
        idtrain = 1:ntrain
        Apm = gen_constraints(Xtrain, ytrain)  #constraint matrix
        return new(Xtrain, ytrain, ntrain, size(Xtrain, 2), Apm)
    end
end

struct testingData <: TestingData
    X::Array{Float64, 2} 
    y::Array{Float64, 1}
    n::Int64
    d::Int64 
    function testingingData(X, y) 
        @assert size(x, 1) == length(y)
        return new(X, y, size(X, 1), size(X, 2))
    end
end

function gen_constraints(X, y)
    n, d = size(X)
    Apm = Array{Int64, 2}(undef, n, n)
    R = CartesianIndices(Apm)
    for I in R 
        i, j = Tuple(I)
        if i == j
            constraint = 0
        elseif y[i] == y[j]
            constraint = 1
        else
            constraint = -1
        end
        Apm[I] = constraint
    end
    return Symmetric(Apm)
end