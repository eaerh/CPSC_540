# Load X and y variable


using JLD
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
include("leastSquares.jl")
model = leastSquares(X,y)


# Report the error on the test set
using Printf
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("TestError = %.2f\n",testError)

# Plot model


using PyPlot


figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((-300,400))
show()

# Fit least squares with RBF
include("leastSquaresRBFL2.jl")
lambda = 0.10
sigma = 0.61

model2 = leastSquaresRBFL2(X, y, lambda, sigma)

# Report the error on the test set

using Printf
n = size(Xtest,1)

Xdist = distancesSquared(Xtest,X)
Gtest = rbfBasis(lambda, sigma, Xdist)


Ztest =  [ Xtest Gtest]


t = size(Ztest,1)
yhat = model2.predict(Ztest)
testError = sum((yhat - ytest).^2)/t
@printf("TestError = %.2f\n",testError)


# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)

Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
Xdist = distancesSquared(Xhat, X)

Ghat = rbfBasis(lambda, sigma, Xdist)
Zhat =  [ Xhat Ghat]
yhat = model2.predict(Zhat)
plot(Xhat[:],yhat,"r")
ylim((-300,400))
savefig("leastSquaresRBFL2.pdf")
show()


#Finding best sigma, lambda
include("leastSquaresRBFL2.jl")
using Printf
using Random
function crossValidate(X, y)
    n = size(X,1)

    # Random permutation for shuffle
    rng = MersenneTwister(1234);
    rndInd = randperm(n)
    mid = floor(Int, n/2)

    # Random shuffle, using the same permutation to conserve correct labels
    Xshuffle, yshuffle = X[rndInd], y[rndInd]

    # Dividing into train and validation set
    Xtrain, ytrain = Xshuffle[1:mid,:], yshuffle[1:mid]
    Xval, yval = Xshuffle[mid+1:end,:], yshuffle[mid+1:end]

    sigma = range(0,stop=10,length=1000)
    lambda = range(0,stop=100,length=1000)
    maxTestError = Inf #best:83.21

    bestSigma = sigma[1] #best: 0.61
    bestLambda = lambda[1] #best: 0.10 500 simulations
    for s in 2:length(sigma)-1
        for l in 2:length(lambda)-1

            model = leastSquaresRBFL2(Xtrain, ytrain, lambda[l], sigma[s])

            Xdist = distancesSquared(Xval,Xtrain)
            Gtest = rbfBasis(lambda[l], sigma[s], Xdist)
            Ztest =  [ Xval Gtest]

            t = size(Ztest,1)
            yhat = model.predict(Ztest)
            testError = sum((yhat - yval).^2)/t

            if (testError < maxTestError)
                maxTestError = testError
                bestSigma = sigma[s]
                bestLambda = lambda[l]
            end

        end
    end

    @printf("Best sigma = %.2f\n",bestSigma)
    @printf("Best lambda = %.2f\n",bestLambda)
    @printf("Best error = %.2f\n",maxTestError)

end

crossValidate(X,y)
