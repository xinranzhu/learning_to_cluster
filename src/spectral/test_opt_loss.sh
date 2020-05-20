set -x 

julia test_opt_loss.jl --set_range 4 --set_Nsample 1 2>&1 | tee > log_test_optloss_4_1.txt 

# julia test_opt_loss.jl --set_range 4 --set_Nsample 2 2>&1 | tee > log_test_optloss_4_2.txt 

# julia test_opt_loss.jl --set_range 4 --set_Nsample 3 2>&1 | tee > log_test_optloss_4_3.txt 

# julia test_opt_loss.jl --set_range 5 --set_Nsample 1 2>&1 | tee > log_test_optloss_5_1.txt 

# julia test_opt_loss.jl --set_range 5 --set_Nsample 2 2>&1 | tee > log_test_optloss_5_2.txt 

# julia test_opt_loss.jl --set_range 5 --set_Nsample 3 2>&1 | tee > log_test_optloss_5_3.txt 

