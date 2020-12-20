####
# Set up basic test problem for comparing approximate "learning to cluster" approach
# and exact spectral clustering approach.
####
using CSV
df = DataFrame(CSV.File("../../data/datasets/abalone.csv", header = 0))
data = convert(Matrix, df[:,2:8])
label = convert(Array, df[:, 9]) # 1 ~ 29
k = 29
