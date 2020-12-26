include("../../src/clustering/laplacian.jl")
include("../../src/utils/derivative/derivative_checker.jl")
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
