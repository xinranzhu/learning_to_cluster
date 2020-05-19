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

function loadmats()
    A = getBodyWeightedAdj()   #weighted adjacency matrix
    B = removeWeights(A) #unweighted adj matrix
    P = getBodyAttributeAdj()  #adjacency matrix of attributes (recall that graph is edge-attributed)
    Pn = normalizeAttributes(P)
    D =  dropdims(sum(A, dims = 2), dims=2) #Degree matrix
    return (A, B, P, Pn, D) #P and Pn not symmetric, but later when we peel off layers, we ensure they are symmetric
end

"""
Get ith layer of sparse tensor. It is assumed that tensor is stored as a 2D sparse array,
with multi-dimensional array entries
"""
function getLayer(Pn, k)
    m, n = size(Pn)
    Layer = spzeros(m, n)
    rows = rowvals(Pn)
    vals = nonzeros(Pn)
    for j = 1:n
        for i in nzrange(Pn, j)
            row = rows[i]
            Layer[row, j] = Pn[row, j][k]
        end
    end
    return symmetric_sum(Layer) #returns symmetric matrix!
end

"""
b is a 1D array of sparse matrices of size (m, n)
"""
function sparseDot(a, b)
    try 
        @assert length(a) == length(b)
    catch e 
        @info "length a",   length(a)
        @info "length b", length(b)
    end
    m, n = size(b[1])
    res = spzeros(m, n)
    for i = 1:length(a)
        res = res + a[i] * b[i]
    end
    return res
end


function sparseDiag(vec)
    sparse(collect(1:size(vec, 1)), collect(1:size(vec, 1)), vec)
end

"""
Input must be diagonal matrix, takes D^{-1/2}
"""
function invertsqrtDiag(spm)
    B  = similar(spm)
    for i = 1:size(spm, 1)
        B[i, i] = 1/sqrt(spm[i, i])
    end
    return B
end

"""
For Reddit dataset, 
 - dL will be an n x n x 12 tensor
- L will be an n x n matrix, where n = 35776

beta is 1x1 float64
alphas is 1x11 float64
"""
function laplacian_attributed(beta, alphas; if_deriv = true)
    A = getBodyWeightedAdj()
    Pn = normalizeAttributes(getBodyAttributeAdj())
    PLayers = [getLayer(Pn, i) for i =1:length(alphas)]

    dbeta = sparseDot(alphas, PLayers) 

    #L = A + beta * dbeta 
    #dL = cat([dbeta], dalphas, dims = 1)

    augmentedA = A + beta * dbeta 
    augmentedD = sparseDiag(dropdims(sum(augmentedA, dims = 2), dims=2))
    invsqrtD = invertsqrtDiag(augmentedD)
    L = invsqrtD * augmentedA * invsqrtD    
    if if_deriv
        dL = []
        dalphas = beta * [PLayers[i] for i = 1:length(alphas)] #dAs
        daugmentedA = cat([dbeta], dalphas, dims = 1)
        daugmentedD1 = sparseDiag(dropdims(sum(dbeta, dims = 2), dims=2))
        daugmentedD2 = beta * [sparseDiag(dropdims(sum(PLayers[i], dims=2), dims=2)) for i =1:length(alphas)]
        dD = cat([daugmentedD1], daugmentedD2, dims=1)
        for i = 1:(1+length(alphas))
            push!(dL, invsqrtD * daugmentedA[i] * invsqrtD + 2 * dD[i] * (-0.5) * (invsqrtD ^ 3) * A * invsqrtD) #derivative of D^{-1/2}AD^{-1/2}
        end 
    else
        dL = nothing
    end
    # have computed A, D, dA, dD
    # L = D^{-1/2}AD^{-1/2
    return L, dL
end

function laplacian_attributed_L(params; if_deriv = true)
    L, dL = laplacian_attributed(params[1], params[2:end]; if_deriv = if_deriv)
    return L, dL
end

