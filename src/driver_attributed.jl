using DataFrames
using CSV
using Latexify

title = CSV.read("../datasets/soc-redditHyperlinks-title.tsv")
body = CSV.read("../datasets/soc-redditHyperlinks-body.tsv")

#pre-processing
subreddits_from = Set(title[1])
subreddits_to = Set(title[2])

const n_title = size(title, 1)
const n_body = size(body, 1)
#571927 total directed edges
#43695 subreddits

# numchar numchar_no_space frac_alphabet frac_digits frac_uppercase frac_white_spaces frac_special_chars num_words num_unique_words num_long_wordsy


function prop_to_mat(vecs_str, num)
    A = Array{Float64, 2}(undef, num - 1, 11);
    for i =1:num-1
        vecs = split(vecs_str[i], ",");
        for j=1:11
            A[i, j] = parse(Float64, vecs[j]) 
        end
    end
    return A
end

title_prop_array = prop_to_mat(title[6], n_title);
title_properties = convert(DataFrame, title_prop_array); #text properties of source post of titles

body_prop_array = prop_to_mat(body[6], n_body);
body_properties = convert(DataFrame, body_prop_array); #text properties of source post of bodies

function latexify_df(df_descr)
    x = describe(df_descr)
    y = convert(Matrix, x)
    y1 = y[:, 2:5]
    y2 = convert(Array{Float32, 2}, y[:, 2:5])
    l = latexify(y2)
end
