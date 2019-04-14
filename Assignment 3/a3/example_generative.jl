using JLD, Printf, Statistics, StatsBase
include("misc.jl")
include("gda.jl")
include("tda.jl")
# Load X and y variable
data = load("gaussNoise.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])



yhat = gda(X, y, Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with GDA: %.3f\n", testError)

yhat = tda(X, y, Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with TDA: %.3f\n", testError)


# Fit a KNN classifier
k = 1
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)
