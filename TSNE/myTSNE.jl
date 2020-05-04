
# helper functions

function get_Q(Y)
    n = size(Y, 1)
    D = pairwise(Euclidean(), Y, Y, dims=1)
    Q = zeros(n, n)
    Q = 1 ./ (1 .+ D.^2)
    Q[diagind(Q)] .= 0
    Q ./= sum(Q)
    return Q
end

function get_single_conditional_p(d, σ)
    neg_d_square = -d / (2.0 * σ^2)
    neg_d_square .-= maximum(neg_d_square) # avoid numerical issues on exp
    exp_neg_d_square = exp.(neg_d_square)
    exp_neg_d_square .+= 1e-8 # avoid numerical issues on log
    return exp_neg_d_square / sum(exp_neg_d_square)
end

function get_perp(d, σ)
    p = get_single_conditional_p(d, σ)
    return 2^(-dot(p,log.(2,p)))
end

function binary_search(func, target, atol = 1e-8, 
        l = 1e-5, r = 1000, maxiter = 1000)
    @assert (func(l) <= target && func(r) >= target) "$(func(l)), $(func(r))"
    for i in 1:maxiter
        middle = (l + r) / 2
        temp = func(middle)
        if isapprox(temp, target, atol=atol)
            return middle
        end
        
        if temp < target
            l = middle
        else
            r = middle
        end
    end
end

function get_conditional_P(X, target_perp=30)
    D = pairwise(Euclidean(), X, X, dims=1).^2
    N = size(X, 1)
    P = zeros(N, N)
    for i in 1:N
        func(σ) = get_perp(D[i, 1:end .!= i], σ)
        sigma = binary_search(func, target_perp)
        P[i, 1:end .!= i] = 
            get_single_conditional_p(D[i, 1:end .!= i], sigma)
    end
    return P
end

function get_gradient(P, Q, Y)
    N, d = size(Y)
    inv_d = 1 ./ (pairwise(Euclidean(), Y, Y, dims=1).^2 .+ 1)
    PQ = P .- Q
    y_diff = reshape(Y, (N, 1, d)) .- reshape(Y, (1, N, d)) # NxNxd
    return 4 .* dropdims(sum(reshape(PQ .* inv_d, (N, N, 1)) .* y_diff, dims=2), dims=2)
end

function plot_points_2D(Y, label, iters)
    plt = plot(legendfont=font(5))
    for i in 1:10
        print()
        yi = Y[label .== i-1, :]
        plt = scatter!(yi[:,1], yi[:,2], label = "$i")
    end
    title!("TSNE on MNIST: Iteration $iters")
    display(plt)
    savefig("figures/MNIST_$iters.pdf") 
end

function mytsne(X, d; 
                perp = 30.0, perp_tol = 1e-5, maxiter = 1000,
                early_exag = true, checkpoint_every=100, lr=100,
                calculate_error_every =100, plot_every = 100)
    # number of training points
    N = size(X, 1)
    P = get_conditional_P(X, perp)
    P = (P .+ P') ./ (2 * N)
    P = max.(P, 1e-12)
    if early_exag
        P .*= 4
    end
    distY = MvNormal(zeros(d), I(d).*1e-4)
    Y = rand!(MersenneTwister(1234), distY, zeros(d, N))'
    dY = zeros(N, d)
    gains = ones(N, d)
    min_gain = 0.01
    checkpoint_Y = []
    # plot_points_2D(Y, label, 0)
    for t in 1:maxiter
        Q = get_Q(Y)
        gradient = get_gradient(P, Q, Y)
        alpha =  t < 250 ? 0.5 : 0.8
        gains = (gains .+ 0.2) .* ((gradient .> 0.) .!= (dY .> 0.)) .+ (gains .* 0.8) .* ((gradient .> 0.) .== (dY .> 0.))
        gains[gains .< min_gain] .= min_gain

        # update stepsize
        dY = - lr .* gains .* gradient .+ alpha .* dY
        Y = Y .+ dY

        if early_exag && t == 50
            P = P ./ 4
        end


        if t % checkpoint_every == 0
            append!(checkpoint_Y, copy(Y))
        end

        if plot_every > 0 && t % plot_every == 0
            plot_points_2D(Y, label, t)
        end

        if t % calculate_error_every == 0
            Q = max.(Q, 1e-12)
            C = sum(P .* log.(P ./ Q))
            println("Iteration $(t + 1): loss $C")
        end
    end
    return Y
end        