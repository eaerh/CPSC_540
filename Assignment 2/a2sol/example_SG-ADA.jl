# Load X and y variable
using JLD, Printf, LinearAlgebra
data = load("quantum.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
lambda = 1
delta = 1

# Initialize
maxPasses = 10
progTol = 1e-4
verbose = true
w = zeros(d,1)
lambda_i = lambda/n # Regularization for individual example in expectation
gradRecord = zeros(d)
D = zeros((d,d))

# Start running stochastic gradient
w_old = copy(w);
for k in 1:maxPasses*n
    global gradRecord
    global delta
    global D
    
    # Choose example to update 'i'
    i = rand(1:n)
    
    # Compute gradient for example 'i'
    r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
    g_i = r_i*X[i,:] + (lambda_i)*w
    gradRecord += g_i.^2
    
    for j in 1:d
        D[j,j] = 1/sqrt(delta + gradRecord[j])
    end
    

    # Choose the step-size
    alpha = 1/(k^(1/2)) # 1/(lambda_i*k)

    # Take thes stochastic gradient step
    global w -= alpha*D*g_i

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
