include("../../src/clustering/laplacian.jl")
include("../../src/clustering/spectral.jl")
include("../../src/utils/derivative/derivative_checker.jl")
using Plots
x = [1 2.0 1.0; 3 4.1 1.2; 2.1 2.3 1.1]
(a,b) = affinity_A(x, [1.0;0.5;1.2], derivative = true)

## Affinity matrix (Spectral clustering paper by Ng and Jordan) derivative test
# entries are Gaussian
f = y->affinity_A(x, y, derivative = true)[1][:]
df = y->affinity_A(x, y, derivative = true)[2][:]
#df([2;3.0])[:]
(r1, r2, r3, r4) = checkDerivative(f, df, [1.1, 0.5, 0.4])
r4

## Laplacian derivative test
g = y->laplacian_L(x, y, derivative = true)[1][:]
dg = y->laplacian_L(x, y, derivative = true)[2][:]
dg([1.1, 1.1, 1.1])

(r1, r2, r3, r4) = checkDerivative(g, dg, [1.1, 0.1, 0.5], nothing, 1, 7)
r4

## Eigenvector derivative test
x = [1 2.0 1.0; 3 4.1 1.2; 2.1 2.3 1.1]
n = size(x, 1)
k=2 #number of eigenvectors to compute
"""
Computes first k eigenvectors of Laplacian matrix of x
"""
function h(theta)
    (L, dL) = laplacian_L(x, theta, derivative = true)
    ef = eigen(Symmetric(L), n-k+1:n)
    V = ef.vectors
    Λ = ef.values
    return V[:]
end

function dh(theta)
    (L, dL) = laplacian_L(x, theta, derivative = true)
    ef = eigen(Symmetric(L), n-k+1:n)
    V = ef.vectors
    Λ = ef.values
    return comp_dV(V, Λ, L, dL, length(theta))[:]
end
h([1.0, 2.0, 3.0])
dh([1.0, 2.0, 3.0])
(r1, r2, r3, r4) = checkDerivative(h, dh, [1.1, 2.1, 0.5], nothing, 1, 7)
r4[1]
@test r4[1] ≈ 2.0 atol = 0.5

## Normalized eigenvector derivative test

x = rand(50, 20)
n = size(x, 1)
d = size(x, 2)
k=10 #number of eigenvectors to compute

function c(theta)
    (L, dL) = laplacian_L(x, theta, derivative = true)
    V , Λ = compute_eigs(Matrix(L), k) #compute leading eigs
    if true
        rownorms = mapslices(norm, V; dims = 2)
        V = broadcast(/, V, rownorms)
    end
    return V[:]
end

function dc(theta)
    (L, dL) = laplacian_L(x, theta, derivative = true)
    V , Λ = compute_eigs(Matrix(L), k) #compute leading eigs
    return comp_dV(V, Λ, L, dL, length(theta); normalized = true)[:]
end

c(ones(d))
dc(ones(d))
(r1, r2, r3, r4) = checkDerivative(c, dc, 3*ones(d), nothing, 10, 15, 20)
r4

r3
