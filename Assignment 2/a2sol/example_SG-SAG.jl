# Load X and y variable
using JLD, Printf, LinearAlgebra
data = load("quantum.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
lambda = 1

# Initialize
maxPasses = 10
progTol = 1e-4
verbose = true
w = zeros(d,1)
lambda_i = lambda/n # Regularization for individual example in expectation

# Memory for gradients
g = zeros(d)  # effective gradient
V = zeros((d,n))  # matrix containing memory of v_i

# Computing Lipschitz-constant
(n, d) = size(X)
norms = [norm(X[i,:], 2)^2 for i in 1:n]
L = 0.25*maximum(norms) + lambda_i

# Start running stochastic gradient
w_old = copy(w);
for k in 1:maxPasses*n
    global g

    # Choose example to update 'i'
    i = rand(1:n)

    # Compute gradient for example 'i'
    r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
    g_i = r_i*X[i,:] + (lambda_i)*w
    
    g = g - V[:, i] + g_i
    V[:, i] = g_i

    # Choose the step-size
    alpha =  1/L 

    # Take thes stochastic gradient step
    global w -= alpha*g/n

    # Check for lack of progress after each "pass"
    if mod(k,n) == 0
        yXw = y.*(X*w)
        f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        global w_old = copy(w);
    end
end
