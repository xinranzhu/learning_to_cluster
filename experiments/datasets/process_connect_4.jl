using CSV
using JLD
using DataFrames
df = DataFrame(CSV.File("experiments/datasets/connect-4.csv", header = 0))
original_data = convert(Matrix, df[1:67557,1:42])
original_label = convert(Array, df[1:67557, 43]) # 1 ~ 29

#process labels
label = zeros(length(original_label))
d = Dict("win"=>1, "loss"=>2, "draw"=>3)
for i = 1:length(label)
    label[i] = Int(d[original_label[i]])
end

#process data
d2 = Dict("x"=>1, "o"=>-1, "b"=>0)
data = zeros(size(original_data, 1), size(original_data, 2))
for i = 1:size(data, 1)
    for j = 1:size(data, 2)
        data[i,j] = d2[original_data[i, j]]
    end
end

save("connect-4.jld", "data", data, "label", label, "k", 3, "date", "12_31_20")
