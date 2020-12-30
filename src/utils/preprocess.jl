using CSV
using JLD
using DataFrames
df = DataFrame(CSV.File("experiments/datasets/abalone.csv", header = 0))
data = convert(Matrix, df[1:4177,2:8])
label = convert(Array, df[1:4177, 9]) # 1 ~ 29
# relabel: regroup labels <= 5 as one lable, and >=15 as one label
# then target number of clusters = 11
#skip = 6, 8, 10, 12, 14
function foo(x)
    #if x in [4, 5, 8, 9, 12, 13]
    if x in [9, 10]
        return true
    else
        return false
    end
end
#filter out select labels
inds = findall(foo, label)
complement = [i for i in 1:length(label) if i ∉ inds]
deleteat!(label, inds)
data = data[complement, :]
#group labels
label[label .< 9] .= 1
#label[(&).(label .> 10, label .<12)] .= 2
label[label .> 10] .= 2
k = 2

save("abalone_2_class.jld", "data", data, "label", label, "k", k, "desc", "leave out labels 9 and 10, split at these values", "date", "12_29_20")
#load
