using DataFrames
using CSV
using Latexify
using SparseArrays

title = CSV.read("../../data/soc-redditHyperlinks-title.tsv")
body = CSV.read("../../data/soc-redditHyperlinks-body.tsv")

#pre-processing
subreddits_from = Set(body[1])
subreddits_to = Set(body[2])

const n_title = size(title, 1)
const n_body = size(body, 1)
#571927 total directed edges
#43695 subreddits

"""
Get dictionary of subreddits, where keys are numbers and values are strings
"""
function getDict(df)
    subreddits_from = df[:, 1]
    subreddits_to = df[:, 2]
    b = Set(subreddits_from);
    e = Set(subreddits_to);
    f = union(b, e);
    len = length(f);
    t = 1:len;
    c = zip(t, f);
    subreddits = Dict{Int, String}()
    for pair in c
        push!(subreddits, pair[1] => pair[2])
    end
    return subreddits
end


# numchar numchar_no_space frac_alphabet frac_digits frac_uppercase frac_white_spaces frac_special_chars num_words num_unique_words num_long_wordsy
"""
"""
function prop_to_mat(vecs_str, num)
    A = Array{Float64, 2}(undef, num, 11);
    for i =1:num
        vecs = split(vecs_str[i], ",");
        for j=1:11
            A[i, j] = parse(Float64, vecs[j]) 
        end
    end
    return A
end

#define dict structures/translation dictionaries
num2str = getDict(body)
str2num = Dict(value => key for (key, value) in num2str)

#properties
title_prop_array = prop_to_mat(title[6], n_title);
title_properties = convert(DataFrame, title_prop_array); #text properties of source post of titles

body_prop_array = prop_to_mat(body[6], n_body);
body_properties = convert(DataFrame, body_prop_array); #text properties of source post of bodies

"""
Get numerical equivalents of labels in reddit body data
Return arrays with these integer labels 
"""
function getBodyNumericalLabel()
    from = body[:, 1]
    to = body[:, 2]
    from_str = Array{Int64, 1}(undef, length(from))
    to_str = Array{Int64, 1}(undef, length(to))
    for i = 1:length(from)
        from_str[i] = str2num[from[i]]
        to_str[i] = str2num[to[i]]
    end
    return from_str, to_str
end

"""
Get weighted adjacency matrix for body data
"""
function getBodyWeightedAdj()    
    (from, to) = getBodyNumericalLabel()
    ss = sparse(from, to, ones(length(from))) #sparse adjacency matrix
    return ss
end

"""
Get ordinal attribute adjacency matrix (pool together edge-attributes)
"""
function getBodyAttributeAdj()
    (from, to) = getBodyNumericalLabel()
    x = Array{Array{T, 1} where T<:Real, 1}(undef, size(body_prop_array, 1))
    for i = 1:size(body_prop_array, 1)
        x[i] = body_prop_array[i, :]
    end
    ss = sparse(from, to, x) #sparse adjacency matrix
    return ss
end


#diagonal(ss) = (ss[i, i] for i = 1:size(ss, 1))
#d =diagonal(ss)
#cc = 0;
#for iter in d
#    println(iter)
   # global cc = cc + ss[iter];
#
#println("number nonzero entries of body adj matrix: ", count(!iszero, ss))
#println("number zero of body adj matrix: ", count(iszero, ss))





function latexify_df(df_descr)
    x = describe(df_descr)
    y = convert(Matrix, x)
    y1 = y[:, 2:5]
    y2 = convert(Array{Float32, 2}, y[:, 2:5])
    l = latexify(y2)
end





