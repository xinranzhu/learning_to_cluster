include("readAttributed.jl")

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
D =  dropdims(sum(A, dims = 2), dims=2) #Degree matrix



