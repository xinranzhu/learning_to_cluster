include("../src/l2c.jl")
using Test
#include("../test/kernels/test_kernel.jl")
#cluster(;method = "hi")
#include("")
## Clustering
include("clustering/laplacian.jl")
include("clustering/spectral.jl")
include("clustering/kmeans_match_labels.jl")

## Core
include("core/datastructs.jl")

## Kernels
include("kernels/test_kernel.jl")

## Utils - this test takes a minute or two
#include("utils/derivative/test_derivative_checker.jl")
