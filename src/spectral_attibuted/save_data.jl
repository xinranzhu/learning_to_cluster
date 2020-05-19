using JLD
using LinearAlgebra

# generate and save N choices of range
# range_set = tensor N*12*2
range1 = [1.  10]
range2 = repeat([0. 1.], 11, 1)
range_set = reshape(vcat(range1, range2), 1, 12, 2)
save("./saved_data/reddit_range_set.jld", "data", range_set)

# PS: could also store each range separately, i.e. save into N different files 
#     such that don't have to load whole file only to get one range setting everyime

# save a bunch of sampling numbers
N_sample_set = [100, 200, 500]
save("./saved_data/reddit_Nsample_set.jld", "data", N_sample_set)


# save a bunch of theta 
Nθ = 10; dimθ =  12
rangeθ = reshape(range_set[1, :, :], dimθ, 2)
θ_set = rand(dimθ, Nθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
save("./saved_data/reddit_theta_set.jld", "data", θ_set)

# save some k values (number of clusters)
k_set = [12, 15, 18, 20, 25, 30, 35, 40]
save("./saved_data/reddit_k_set.jld", "data", k_set)

