function gen_constraints(X, y)
    n, d = size(X)
    Apm = Array{Float64, 2}(undef, n, n)
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