include("readAttributed.jl")

"""
Normalize the edge attributes of A (stored in P) to be in [0, 1]
"""
function normalizeAttributes(P)
    Pn = similar(P)
    maxs = zeros(11)
    rows = rowvals(P)
    vals = nonzeros(P)
    m, n = size(P)
    for j = 1:n
        for i in nzrange(P, j)
            val = vals[i]
            for k = 1:length(val)
                if val[k] > maxs[k]
                    maxs[k] = val[k]
                end
            end
        end
    end
    for j = 1:n
        for i in nzrange(P, j)
            val = vals[i]
            row = rows[i]
            Pn[row, j] = P[row, j] ./ maxs
        end
    end
    return Pn
end


function removeWeights(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    B  = similar(A)
    for j = 1:n
        #println("nzrange: ", nzrange(A, j))
    for i in nzrange(A, j)
        row = rows[i]
        #val = vals[i]
        #println("row: ", row)
        #println("val: ", val)
        #println("ACTUAL VAL:", A[row, j])
        # perform sparse wizardry...
        B[row, j] = 1
    end
    end
    return B
end

A = getBodyWeightedAdj()   #weighted adjacency matrix
B = removeWeights(A) #unweighted adj matrix
P = getBodyAttributeAdj()  #adjacency matrix of attributes (recall that graph is edge-attributed)
Pn = normalize(P)
D =  dropdims(sum(A, dims = 2), dims=2) #Degree matrix




