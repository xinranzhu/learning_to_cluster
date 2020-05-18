using Distributions
using TensorOperations
using MLDatasets
using Random
using DataFrames
using CSV

# include("comp_deriv.jl")
# include("../kernels/kernels.jl")
include("spectral_reduction_main.jl")

# MNIST
# dataraw, _ = MNIST.testdata(Float64)
# data = reshape(permutedims(dataraw,(3, 1, 2)), size(dataraw, 3), size(dataraw,1)*size(dataraw,2));

# abalone
df = DataFrame(CSV.File("../datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8]) 

# shuffle data: n * d
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1))
data = data[ind_shuffle, :]

###########################################################
X = data[1:4177, :]; n, d = size(X)
ntest = 100; h = 1e-5
I_rows = nothing
# I_rows = randperm(n)[1:Int(floor(n/2))]

fun_A(θ) = affinity_A(X, θ; if_deriv = false)[1]
deriv_A(θ) =   affinity_A(X, θ)[2]

# 1d parameter
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dAtest_fd = deriv_fd(fun_A, h)
θgrid = range(1, stop=2000, length=ntest)
# θgrid = range(0.01, stop=2, length=ntest)
err1 = maximum(norm.(deriv_A.(θgrid) - dAtest_fd.(θgrid))) # O(h^2)
@info "dA, One-dim parameter: " err1

# multi-dimensional parameter
n, d = size(X)
n1 = I_rows == nothing ? n : length(I_rows)
θgrid = rand(Uniform(1,2000), d, ntest)
# θgrid = rand(Uniform(0.01,2), d, ntest)
dAh = Array{Float64, 2}(undef, n1, n)
err2 = 0.
hvec = h .* ones(d)
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    @tensor dAh[l, j] = deriv_A(θ)[l, j, k] * hvec[k]
#     err_current = norm(fun_A(θ .+ h) .- fun_A(θ) - dAh) # O(h^2)
    err_current = norm(fun_A(θ .+ h) - fun_A(θ .- h)- 2 .* dAh) # O(h^3)
#     @info err_current
    err2 = max(err2, err_current)
end
@info "dA, Multidimensional parameter: " err2


############################################################
fun_D(θ) = degree_D(X, θ; if_deriv = false)[1]
deriv_D(θ) =  degree_D(X, θ)[2]

# 1d parameter
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h # O(h^2)
dDtest_fd = deriv_fd(fun_D, h)
θgrid = range(1, stop=2000, length=ntest)
# θgrid = range(0.01, stop=2, length=ntest)
err1 = maximum(norm.(deriv_D.(θgrid) - dDtest_fd.(θgrid))) # O(h^2)
@info "dD, One-dim parameter: " err1

# multi-dimensional parameter
θgrid = rand(Uniform(1,2000), d, ntest)
# θgrid = rand(Uniform(0.01,2), d, ntest)
dDh = Array{Float64, 2}(undef, n1, n)
err2 = 0.
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    dDh = deriv_D(θ) * hvec
    err_current = norm(fun_D(θ .+ h) - fun_D(θ) - dDh)  # O(h^2)
#     err_current = norm(fun_D(θ .+ h) - fun_D(θ .- h) - 2 .* dDh) # O(h^3)
#     @info err_current
    err2 = max(err2, err_current)
end
@info "dD, Multidimensional parameter: " err2

############################################################

fun_L(θ) = laplacian_L(X, θ; I_rows = I_rows, if_deriv = false)[1]
deriv_L(θ) =  laplacian_L(X, θ; I_rows = I_rows)[2]

# 1d parameter
deriv_fd(f, h) = x -> (f(x+h)-f(x-h))/2/h
dLtest_fd = deriv_fd(fun_L, h)
θgrid = range(1, stop=2000, length=ntest)
# θgrid = range(0.01, stop=2, length=ntest)
# err1 = maximum(norm.(deriv_L.(θgrid) - dLtest_fd.(θgrid))) # O(h^2)
err1 = norm(fun_L.(θgrid .+ h) - fun_L.(θgrid) - h .* deriv_L.(θgrid)) # O(h^2)
# err1 = norm(fun_L.(θgrid .+ h) - fun_L.(θgrid .- h) - 2 * h .* deriv_L.(θgrid)) # O(h^3)
@info "dL, One-dim parameter: " err1

# multi-dimensional parameter
θgrid = rand(Uniform(1, 2000), d, ntest)
# θgrid = rand(Uniform(0.01,2), d, ntest)
dLh = Array{Float64, 2}(undef, n1, n)
err2 = 0.
for i in 1:ntest
    global err2
    θ = θgrid[:, i]
    @tensor dLh[l, j] = deriv_L(θ)[l, j, k] * hvec[k]
    err_current = norm(fun_L(θ .+ h) - fun_L(θ) - dLh) # O(h^2)
#     err_current = norm(fun_L(θ .+ h) - fun_L(θ .- h) - 2 .* dLh) # O(h^3)
#     @info err_current
    err2 = max(err2, err_current)
end
@info "dL, Multidimensional parameter: " err2

################################################################