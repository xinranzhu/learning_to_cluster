using LinearAlgebra
using Plots
using MLDatasets
using Statistics
using Random
using Distances
using Distributions
using Roots
using ArgParse
using Clustering
using Combinatorics
using ProgressMeter
using Dates
using Printf
import Sobol: next!
using Sobol
using DataFrames
using CSV
using TensorOperations
using JLD
using Optim
using Arpack

include("../attributed/attributed.jl")
include("../datastructs.jl")
include("model_reduction.jl")

s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--set_range"
        help = "use which settings of rangeθ -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--set_Nsample"
        help = "use which settings of N_sample -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--set_k"
        help = "use which settings of k -- see precompute_Vhat"
        arg_type = Int
        default = 1
    "--idtheta"
        help = "assigned value of theta, skip training -- should be load from file"
        arg_type = Int
        default = 0
    "--which"
        help = "which derivative to test"
        arg_type = String
        default = "L"
end
parsed_args = parse_args(ARGS, s)

# load Vhat 
Vhat_set = JLD.load("./saved_data/Vhat_set_now.jld")["data"]
Vhat = Vhat_set.Vhat
rangeθ = Vhat_set.rangeθ
k = Vhat_set.k
d = size(rangeθ, 1)
m = size(Vhat, 2)

@info "Finish loading Vhat, k=$k, dtheta = $d"
n = 35776
(idtrain, ytrain) = trainInfo_fixed()
ntrain = length(idtrain)
traindata = atttraindata(ntrain, idtrain, ytrain)
@info "Finish build training data, ntrain=$ntrain, type of traindata = $(typeof(traindata))."

ntest = 10; h = 1e-5; hvec = h * ones(d)
# use quasi-random theta samples
s = SobolSeq(rangeθ[:,1], rangeθ[:,2])
N = hcat([next!(s) for i = 1:ntest]...)' # ntest * d
@info "Size of testing theta grid: $(size(N))"

function comp_dY_full(k, θ, Vhat; if_deriv = true)
    dimθ = length(θ)
    n, m = size(Vhat)
    L, dL = laplacian_attributed_L(θ; if_deriv = true)
    H = @views Vhat' * L * Vhat 
    ef = eigen(Symmetric(H), m-k+1:m)
    Y = ef.vectors
    Λ = ef.values
    @assert size(Y) == (m, k)
    @assert length(Λ) == k
    if if_deriv
        dH = Array{Float64, 3}(undef, m, m, dimθ)
        for i in 1:dimθ
            dH[:,:,i] = Vhat' * dL[i] * Vhat
        end
        dY = comp_dY(Y, Λ, H, dH, dimθ)
    else
        dY = nothing
    end
    return Y, dY
end

fun_L(θ) = laplacian_attributed_L(θ; if_deriv = false)[1] # takes 45s
deriv_L(θ) = laplacian_attributed_L(θ; if_deriv = true)[2] # takes 45s
fun_Y(θ) = comp_dY_full(k, θ, Vhat; if_deriv = false)[1]
deriv_Y(θ) =  comp_dY_full(k, θ, Vhat; if_deriv = true)[2]
loss(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat; if_deriv = false)[1] 
loss_deriv(θ) = loss_fun_reduction(n, k, θ, traindata, Vhat)[2] 

dYh = Array{Float64, 2}(undef, m, k)
err = 0.
before = Dates.now()
for i = 1:ntest
    θ = N[i, :]
    global err
    if parsed_args["which"] == "L"
        dL = deriv_L(θ) 
        dLh = dL[1] .* hvec[1]
        for i in 2:12
            dLh += dL[i] .* hvec[i]
        end
        err_current = norm(fun_L(θ .+ h) - fun_L(θ) - dLh) # O(h^2)
    elseif parsed_args["which"] == "Y"
        @tensor dYh[l, j] = deriv_Y(θ)[l, j, k] * hvec[k]
        err_current = norm(fun_Y(θ .+ h) - fun_Y(θ) - dYh) # O(h^2)
    else
        @info "Testing derivative of loss function."
        dlh = dot(loss_deriv(θ), hvec)
        err_current = norm(loss(θ .+ h) - loss(θ) - dlh) # O(h^2)
    end
    @info "$i test: current error = $err_current"
    err = max(err, err_current)
end
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5) 
@info "Test derivative of $(parsed_args["which"]): error = $err. Each evaluation took $(elapsedmin/ntest) min." 

# TEST dL
# [ Info: Finish loading Vhat, k=12, dtheta = 12
# [ Info: Finish build training data, ntrain=1662, type of traindata = atttraindata.
# [ Info: Size of testing theta grid: (10, 12)
# [ Info: 1 test: current error = 9.31554016244049e-6
# [ Info: 2 test: current error = 1.3994456020554494e-5
# [ Info: 3 test: current error = 5.528957588712038e-6
# [ Info: 4 test: current error = 7.766985172142915e-6
# [ Info: 5 test: current error = 1.5499407596531632e-5
# [ Info: 6 test: current error = 1.0832423798781278e-5
# [ Info: 7 test: current error = 4.12925403541534e-6
# [ Info: 8 test: current error = 5.027630667461057e-6
# [ Info: 9 test: current error = 1.266557873773595e-5
# [ Info: 10 test: current error = 1.5718143542297014e-5
# [ Info: Test derivative of L: error = 1.5718143542297014e-5. Each evaluation took 2.2673069999999997min.

# TEST dY
# [ Info: Finish loading Vhat, k=12, dtheta = 12
# [ Info: Finish build training data, ntrain=1662, type of traindata = atttraindata.
# [ Info: Size of testing theta grid: (10, 12)
# [ Info: 1 test: current error = 3.1389816775722693e-6
# [ Info: 2 test: current error = 7.90606777629479e-6
# [ Info: 3 test: current error = 6.100639834114737e-7
# [ Info: 4 test: current error = 2.3512693542454628e-6
# [ Info: 5 test: current error = 8.683180546303226e-6
# [ Info: 6 test: current error = 3.327913581898069e-6
# [ Info: 7 test: current error = 3.68048062806304e-7
# [ Info: 8 test: current error = 7.492284129219651e-7
# [ Info: 9 test: current error = 6.487791687746845e-6
# [ Info: 10 test: current error = 7.562903872085179e-6
# [ Info: Test derivative of Y: error = 8.683180546303226e-6. Each evaluation took 2.309303 min.

#TEST dloss
# [ Info: Test derivative of l: error = 0.22003529363259464. Each evaluation took 2.238413 min.