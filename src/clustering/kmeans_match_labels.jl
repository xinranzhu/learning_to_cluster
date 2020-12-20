using Hungarian

function mode_match_labels(kmeans_labels, actual_labels)
    inferred_labels = Dict{Int64, Int64}()
    for i in 1:10
        index = findall(x -> x == i, kmeans_labels)
        mode_label = mode(actual_labels[index])
        push!(inferred_labels, i => mode_label)
    end
    return inferred_labels
end

function ACC_match_labels(assignment, actual_labels, k, dataset)
    current_acc = 0.
    current_matched_assignment = nothing
    if k > 10
        @warn "To many clusters. Can't do brute force label matching"
    end
    for perm in permutations(1:k)
        mapping(i) = perm[i]
        matched_assignment = mapping.(assignment)
        next_acc = mean(matched_assignment .== actual_labels)
        if next_acc > current_acc  
            current_acc = next_acc
            current_matched_assignment = matched_assignment
        end
    end
    return current_acc, current_matched_assignment
end

function bipartite_match_labels(assignment, actual_labels, k; trainmax = 0)
    # how many nodes in cluster i have true label j
    # cost of assigning i to label j
    function cost(i, j)
        idx_i = findall(x->x!=i, assignment)
        idx_j = findall(x->x==j, actual_labels)
        cost_ij = length(intersect(idx_i, idx_j))
        return cost_ij
    end
    weights = Array{Float64, 2}(undef, k, k)
    for i in 1:k, j in 1:k 
        weights[i, j] = cost(i, j)
    end
    mapping, mincost = hungarian(weights)
    mapping_fun(i) = mapping[i]
    bip_assignment = mapping_fun.(assignment)
    acc = mean(bip_assignment[trainmax+1:end] .== actual_labels[trainmax+1:end])
    return acc, bip_assignment
end

