set -x 

julia test_opt_loss.jl --set_range 1 --set_Nsample 1 > log_test_optloss_1_1.txt 2>&1

julia test_opt_loss.jl --set_range 1 --set_Nsample 2 > log_test_optloss_1_2.txt 2>&1

julia test_opt_loss.jl --set_range 1 --set_Nsample 3 > log_test_optloss_1_3.txt 2>&1

julia test_opt_loss.jl --set_range 2 --set_Nsample 1 > log_test_optloss_2_1.txt 2>&1

julia test_opt_loss.jl --set_range 2 --set_Nsample 2 > log_test_optloss_2_2.txt 2>&1

julia test_opt_loss.jl --set_range 2 --set_Nsample 3 > log_test_optloss_2_3.txt 2>&1
