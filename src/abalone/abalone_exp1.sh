set -x 
# 05/16/17:55
# julia Exp_script.jl --dataset "abalone"  --ntotal 4177 > log_exp_abalone_full_kmeans.txt 2>&1

# julia Exp_script.jl --dataset "abalone"  --ntotal 4177 --TSNE > log_exp_abalone_full_tsne.txt 2>&1

# julia Exp_script.jl --dataset "abalone"  --ntotal 4177 --spectral --specparam 0 --single > log_exp_abalone_full_spectral_single.txt 2>&1

# julia Exp_script.jl  --ntotal 4177 --spectral --specparam 0 --single --rangetheta 0.1 200. > log_exp_abalone_full_spectral_single_range1.txt 2>&1

# julia Exp_script.jl  --ntotal 4177 --spectral --specparam 0 --single --rangetheta 0.1 100. > log_exp_abalone_full_spectral_single_range2.txt 2>&1

# julia Exp_script.jl  --ntotal 4177 --spectral --specparam 0 --single --rangetheta 1 200. > log_exp_abalone_full_spectral_single_range4.txt 2>&1

# julia Exp_script.jl  --ntotal 4177 --spectral --specparam 0 --single --rangetheta 0.1 1000. > log_exp_abalone_full_spectral_single_range5.txt 2>&1

# julia Exp_script.jl  --ntotal 4177 --spectral --specparam 0 --single --rangetheta 0.1 2000. > log_exp_abalone_full_spectral_single_range6.txt 2>&1

# 05/17/17:55
# julia exp_script.jl  --spectral --specparam 0 --single --set_range 2 > log_exp_abalone_spectral_single_range2.txt 2>&1

# julia exp_script.jl  --spectral --specparam 0 --single --set_range 3 > log_exp_abalone_spectral_single_range3.txt 2>&1

# julia exp_script.jl  --spectral --specparam 0 --single --set_range 6 > log_exp_abalone_spectral_single_range5.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 1 --set_Nsample 1 > log_exp_abalone_spectral_reduction_single_range1_N1.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 1 --set_Nsample 2 > log_exp_abalone_spectral_reduction_single_range1_N2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 1 --set_Nsample 3 > log_exp_abalone_spectral_reduction_single_range1_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 1 --set_Nsample 4 > log_exp_abalone_spectral_reduction_single_range2_N4.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 2 --set_Nsample 1 > log_exp_abalone_spectral_reduction_single_range2_N1.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 2 --set_Nsample 2 > log_exp_abalone_spectral_reduction_single_range2_N2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 2 --set_Nsample 3 > log_exp_abalone_spectral_reduction_single_range2_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 2 --set_Nsample 4 > log_exp_abalone_spectral_reduction_single_range2_N4.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 3 --set_Nsample 1 > log_exp_abalone_spectral_reduction_single_range3_N1.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 3 --set_Nsample 2 > log_exp_abalone_spectral_reduction_single_range3_N2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 3 --set_Nsample 3 > log_exp_abalone_spectral_reduction_single_range3_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 3 --set_Nsample 4 > log_exp_abalone_spectral_reduction_single_range3_N4.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 > log_exp_abalone_spectral_reduction_single_range4_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 > log_exp_abalone_spectral_reduction_single_range4_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 > log_exp_abalone_spectral_reduction_single_range4_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 > log_exp_abalone_spectral_reduction_single_range4_N4.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 > log_exp_abalone_spectral_reduction_single_range5_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 > log_exp_abalone_spectral_reduction_single_range5_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 > log_exp_abalone_spectral_reduction_single_range5_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 > log_exp_abalone_spectral_reduction_single_range5_N4.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 6 --set_Nsample 1 > log_exp_abalone_spectral_reduction_single_range6_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 6 --set_Nsample 2 > log_exp_abalone_spectral_reduction_single_range6_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 6 --set_Nsample 3 > log_exp_abalone_spectral_reduction_single_range6_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 6 --set_Nsample 4 > log_exp_abalone_spectral_reduction_single_range6_N4.txt 2>&1

# do only [1, 1000] and [1, 200] range; all N samples

# 29 clusters, only evaluate the testing set (size 3177)
# julia exp_script.jl  > log_exp_abalone_kmeans.txt 2>&1
# julia exp_script.jl --TSNE > log_exp_abalone_TSNE.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1  > log_exp_abalone_reduction_single_range4_N1_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2  > log_exp_abalone_reduction_single_range4_N2_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3  > log_exp_abalone_reduction_single_range4_N3_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4  > log_exp_abalone_reduction_single_range4_N4_p2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1  > log_exp_abalone_reduction_single_range5_N1_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2  > log_exp_abalone_reduction_single_range5_N2_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3  > log_exp_abalone_reduction_single_range5_N3_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4  > log_exp_abalone_reduction_single_range5_N4_p2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --trainsize 400 > log_exp_abalone_reduction_single_range4_N1_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --trainsize 400 > log_exp_abalone_reduction_single_range4_N2_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --trainsize 400 > log_exp_abalone_reduction_single_range4_N3_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --trainsize 400 > log_exp_abalone_reduction_single_range4_N4_p3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --trainsize 400 > log_exp_abalone_reduction_single_range5_N1_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --trainsize 400 > log_exp_abalone_reduction_single_range5_N2_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --trainsize 400 > log_exp_abalone_reduction_single_range5_N3_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --trainsize 400 > log_exp_abalone_reduction_single_range5_N4_p3.txt 2>&1

# adjust loss fun 
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --C 10 > log_exp_abalone_reduction_single_range4_N1_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --C 10 > log_exp_abalone_reduction_single_range4_N2_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --C 10  > log_exp_abalone_reduction_single_range4_N3_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --C 10  > log_exp_abalone_reduction_single_range4_N4_p2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --C 10  > log_exp_abalone_reduction_single_range5_N1_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --C 10  > log_exp_abalone_reduction_single_range5_N2_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --C 10  > log_exp_abalone_reduction_single_range5_N3_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --C 10  > log_exp_abalone_reduction_single_range5_N4_p2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range4_N1_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range4_N2_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range4_N3_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range4_N4_p3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range5_N1_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range5_N2_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range5_N3_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --trainsize 400 --C 10 > log_exp_abalone_reduction_single_range5_N4_p3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --C 50 > log_exp_abalone_reduction_single_range4_N1_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --C 50 > log_exp_abalone_reduction_single_range4_N2_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --C 50  > log_exp_abalone_reduction_single_range4_N3_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --C 50  > log_exp_abalone_reduction_single_range4_N4_p2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --C 50  > log_exp_abalone_reduction_single_range5_N1_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --C 50  > log_exp_abalone_reduction_single_range5_N2_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --C 50  > log_exp_abalone_reduction_single_range5_N3_p2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --C 50  > log_exp_abalone_reduction_single_range5_N4_p2.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range4_N1_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range4_N2_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range4_N3_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range4_N4_p3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range5_N1_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range5_N2_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range5_N3_p3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --trainsize 400 --C 50 > log_exp_abalone_reduction_single_range5_N4_p3.txt 2>&1


# try relabel with new Vhat
# julia exp_script.jl --relabel > log_exp_abalone_kmeans.txt 2>&1
# julia exp_script.jl --TSNE --relabel > log_exp_abalone_TSNE.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --relabel > log_exp_abalone_reduction_single_range4_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --relabel > log_exp_abalone_reduction_single_range4_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --relabel > log_exp_abalone_reduction_single_range4_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --relabel > log_exp_abalone_reduction_single_range5_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --relabel > log_exp_abalone_reduction_single_range5_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --relabel > log_exp_abalone_reduction_single_range5_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 1 --relabel > log_exp_abalone_reduction_single_range7_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 2 --relabel > log_exp_abalone_reduction_single_range7_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 3 --relabel > log_exp_abalone_reduction_single_range7_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range4_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range4_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range4_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range4_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range5_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range5_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range5_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range5_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 1 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range7_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 2 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range7_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 3 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range7_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 4 --relabel --trainsize 400 > log_exp_abalone_reduction_single_range7_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range4_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 2 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range4_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 3 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range4_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 4 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range4_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 1 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range5_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 2 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range5_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 3 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range5_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 5 --set_Nsample 4 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range5_N3.txt 2>&1

# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 1 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range7_N1.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 2 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range7_N2.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 3 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range7_N3.txt 2>&1
# julia exp_script.jl --single --reduction --set_range 7 --set_Nsample 4 --relabel --trainsize 800 > log_exp_abalone_reduction_single_range7_N3.txt 2>&1


#####################
# multi-dim param
# julia exp_script.jl --reduction --set_range 4 --set_Nsample 1 --relabel  > log_exp_abalone_reduction_multi_range4_N1.txt 2>&1
# julia exp_script.jl --reduction --set_range 4 --set_Nsample 2 --relabel  > log_exp_abalone_reduction_multi_range4_N2.txt 2>&1
# julia exp_script.jl --reduction --set_range 4 --set_Nsample 3 --relabel  > log_exp_abalone_reduction_multi_range4_N3.txt 2>&1

# julia exp_script.jl --reduction --set_range 5 --set_Nsample 1 --relabel  > log_exp_abalone_reduction_multi_range5_N1.txt 2>&1
# julia exp_script.jl --reduction --set_range 5 --set_Nsample 2 --relabel  > log_exp_abalone_reduction_multi_range5_N2.txt 2>&1
# julia exp_script.jl --reduction --set_range 5 --set_Nsample 3 --relabel  > log_exp_abalone_reduction_multi_range5_N3.txt 2>&1

# julia exp_script.jl --reduction --set_range 6 --set_Nsample 1 --relabel  > log_exp_abalone_reduction_multi_range4_N1.txt 2>&1
# julia exp_script.jl --reduction --set_range 6 --set_Nsample 2 --relabel  > log_exp_abalone_reduction_multi_range4_N2.txt 2>&1



#####################
# Sep 24 2020
julia exp_script.jl --single --reduction --set_range 4 --set_Nsample 1 --relabel 2>&1 | tee a.out_log_exp_abalone_reduction_single_range4_N1
