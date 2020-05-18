using JLD

# generate and save N choices of range
# range_set = tensor N*12*2

save("./saved_data/reddit_range_set.jld", "data", range_set)

# PS: could also store each range separately, i.e. save into N different files 
#     such that don't have to load whole file only to get one range setting everyime

# save a bunch of sampling numbers
N_sample_set = [100, 250, 500, 1000]
save("./saved_data/reddit_Nsample_set.jld", "data", N_sample_set)


# save a bunch of theta 
θ_set = rand(100, 12)
save("./saved_data/reddit_theta_set.jld", "data", N_sample_set)

