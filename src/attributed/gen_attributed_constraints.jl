#include("readAttributed.jl")
###
### extract cores
###

word_list = ["minecraft", "nosleep", "travel", "soccer", "atheism", "parenting", "science", "music", "news", "technology", "woahdude"]
#preprocess by removing subreddits that belong to 2 or more of these categories

function gen_lists(word_list)
    master_list = []
    for word in word_list
        push!(master_list, hcat(word, adjDict[word]))
    end
    return master_list
end

"""
deletes words form list which appear in multiple groups, this way, we get cores of each group, and can be more confident that
they should have distinct labels
"""
function prune_list(word_list)
    list = gen_lists(word_list)
    masterList = reduce(union, list)
    for word in masterList
        bool = [word in list[i] for i =1:length(list)]
        if sum(bool)>1
            for i = 1:length(list)
                list[i] = filter(y -> y!=word, list[i])
            end
        end
    end
    return list
end

a = prune_list(word_list)

function create_row(length, n)
    x = zeros(n)
    for i =1:length
        x[i] = 0
    end
    for i = length+1:n
        x[i] = 1
    end
    return x
end

"""
Return training indices and labels 
"""
function trainInfo(word_list)
    a = prune_list(word_list)
    f = (x, y) -> cat(x, y, dims=1)
    lengths = [length(a[i]) for i=1:length(a)] #lengths of each mini-list
    sums = [sum(lengths[1:i]) for i = 1:length(a)] #get vector of 1s and zeros, sum these to get distinct labels
    k = reduce(f, a) #flat list of all elts 
    n = length(k)
    #y = zeros(1, length(k))
    y = sum([create_row(sums[i], n) for i = 1:length(a)], dims=1)
    indices = zeros(1, length(k))
    for i = 1:length(k)
        indices[i] = str2num[k[i]]     
    end
    return indices, y
end


function trainInfo_fixed()
    word_list = ["minecraft", "nosleep", "travel", "soccer", "atheism", "parenting", "science", "music", "news", "technology", "woahdude"]
    a = prune_list(word_list)
    f = (x, y) -> cat(x, y, dims=1)
    lengths = [length(a[i]) for i=1:length(a)] #lengths of each mini-list
    sums = [sum(lengths[1:i]) for i = 1:length(a)] #get vector of 1s and zeros, sum these to get distinct labels
    k = reduce(f, a) #flat list of all elts 
    n = length(k)
    #y = zeros(1, length(k))
    y = sum([create_row(sums[i], n) for i = 1:length(a)], dims=1)
    indices = zeros(1, length(k))
    for i = 1:length(k)
        indices[i] = str2num[k[i]]     
    end
    return indices, y
end



