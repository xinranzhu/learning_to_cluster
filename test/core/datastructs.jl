using Test
include("../../src/core/datastructs.jl")

X = [1.2 2; 3 4.7; 5.5 6.0]
y = [1, 1, 2]
ntrain = 3

td = trainingData(X, y, ntrain)

m = td.LinkConstraintsMatrix

@test Matrix(m) == [0 1 -1; 1 0 -1; -1 -1 0]
