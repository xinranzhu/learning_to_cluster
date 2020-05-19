include("readAttributed.jl")
include("clusteringAttributed.jl")
include("gen_attributed_constraints.jl")

(indices, y) = trainInfo_fixed()

#L, dL = laplacian_attributed()