# Load X and y variable
using JLD, Printf, LinearAlgebra



data = load("binaryData.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
X = [ones(n,1) X]
d += 1
w = zeros(d,1)
lambda = 1
maxPasses = 500
progTol = 1e-4
verbose = true

## Run and time coordinate descent to minimize L2-regularization logistic loss

# Start timer
time_start = time_ns()

# Compute Lipschitz constant of 'f'
#sd = eigen(X'X)
#L = maximum(sd.values) + lambda;

#compute new lipschitz constant

Lc = maximum(sum(X.^2, dims=1)) + lambda

#@show(Lc)

# Start running coordinate descent
w_old = copy(w);
iterations = 0

r = X*w - y

for k in 1:maxPasses*d

    # Choose variable to update 'j'
    j = rand(1:d)

    # Compute partial derivative 'g_j'
    """
    r = X*w - y
    g = X'*r + lambda * w
    g_j = g[j];
    """


    global r
    g_j = dot(X[:,j],r)


    #Backtracking line search
    alpha = 1/100
    c = 10^0 #Arbitrary constant

    fxk = (1/2)*norm(r)^2 + lambda/2 * norm(w)^2
    e = zeros(d)
    e[j] = 1
    while true

        r_new = r - X[:,j]*alpha * c * g_j
        #Armijo condition
        if ((1/2)*norm(r_new)^2 + lambda/2 * norm(w - alpha * c * g_j * e)^2
            > fxk)
            alpha /= 4
        else
            break;
        end
    end


    # Update variable
    #w[j] -= (1/Lc)*(g_j + lambda*w[j]);


    #Update variable with variying step size alpha
    w[j] -= alpha*(g_j + lambda*w[j]);

    #Update r

    r = r - X[:,j]*(alpha)*g_j
    #r = r - X[:,j]*(1/Lc)*g_j

    # Check for lack of progress after each "pass"
    # - Turn off computing 'f' and printing progress if timing is crucial
    if mod(k,d) == 0
        r = X*w - y
        f = (1/2)norm(r)^2 + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/d,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        global w_old = copy(w);
    end
    global iterations += 1
end

# End timer
@printf("Seconds = %f\n",(time_ns()-time_start)/1.0e9)
@printf("Iterations = %f\n", iterations)
