using JLD
using LinearAlgebra

# generate and save N choices of range
range_set = [0.1 200.;
            0.1 1000; 
            0.1 1500; 
            1 200;
            1 1000
            1 1500;
            0.1 30.]
save("./saved_data/abalone_range_set.jld", "data", range_set)


# save a bunch of sampling numbers
N_sample_set = [100, 500, 1000, 1500]
save("./saved_data/abalone_Nsample_set.jld", "data", N_sample_set)


# save a bunch of theta 
# Nθ = 10; dimθ =  12
# rangeθ = reshape(range_set[1, :, :], dimθ, 2)
# θ_set = rand(dimθ, Nθ) .* (rangeθ[:, 2] .- rangeθ[:, 1]) .+ rangeθ[:, 1]
# save("./saved_data/abalone_theta_set.jld", "data", θ_set)


