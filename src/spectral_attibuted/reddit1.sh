set -x

julia exp_reddit.jl --set_range 1 --set_Nsample 1 --set_k 1 > log_exp_reddit_r1_N1_k1.txt 2>&1
julia exp_reddit.jl --set_range 1 --set_Nsample 1 --set_k 2 > log_exp_reddit_r1_N1_k2.txt 2>&1
julia exp_reddit.jl --set_range 1 --set_Nsample 1 --set_k 3 > log_exp_reddit_r1_N1_k3.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 1 --set_k 4 > log_exp_reddit_r1_N1_k4.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 1 --set_k 5 > log_exp_reddit_r1_N1_k5.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 1 --set_k 6 > log_exp_reddit_r1_N1_k6.txt 2>&1

# julia exp_reddit.jl --set_range 1 --set_Nsample 2 --set_k 1 > log_exp_reddit_r1_N2_k1.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 2 --set_k 2 > log_exp_reddit_r1_N2_k2.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 2 --set_k 3 > log_exp_reddit_r1_N2_k3.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 2 --set_k 4 > log_exp_reddit_r1_N2_k4.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 2 --set_k 5 > log_exp_reddit_r1_N2_k5.txt 2>&1
# julia exp_reddit.jl --set_range 1 --set_Nsample 2 --set_k 6 > log_exp_reddit_r1_N2_k6.txt 2>&1

