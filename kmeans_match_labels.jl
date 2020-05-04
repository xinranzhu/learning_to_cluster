function mode_match_labels(kmeans_labels, actual_labels)
    inferred_labels = Dict{Int64, Int64}()
    for i in 1:10
        index = findall(x -> x == i, kmeans_labels)
        mode_label = mode(actual_labels[index])
        push!(inferred_labels, i => mode_label)
    end
    return inferred_labels
end

function ACC_match_labels(assignment, actual_labels, k)
    current_acc = 0.
    current_matched_assignment = nothing
    matched_actual_labels = label .+ 1 # adjust true labels to range 1:k
    for perm in permutations(1:k)
        mapping(i) = perm[i]
        matched_assignment = mapping.(assignment)
        next_acc = mean(matched_assignment .== matched_actual_labels)
        if next_acc > current_acc  
            current_acc = next_acc
            current_matched_assignment = matched_assignment
        end
    end
    return current_acc, current_matched_assignment, matched_actual_labels
end