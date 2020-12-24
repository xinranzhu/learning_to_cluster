module TypeUtils
    function is_none(x::Any)
        return x == nothing ? true : false
    end
end
include("linear_algebra_utils.jl")
include("derivative/derivative_checker.jl")
