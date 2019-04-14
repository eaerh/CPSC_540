# Load X and y variable
using JLD
data = load("groupData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Include file containing models
include("logReg.jl")


# Fit multi-class logistic regression classifer
model = logRegSoftmax(X,y)

# Compute training and validation error
using Statistics
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),dims=2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);




# Fit multi-class logistic regression classifer with L2-regularization
model = logRegSoftmaxL2(X,y,10)

# Compute training and validation error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),dims=2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);


# Fit multi-class logistic regression classifer with L1-regularization
model = logRegSoftmaxL1(X,y,10)

# Compute training and validation error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),dims=2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);




# Fit multi-class logistic regression classifer with L1 group regularization
model = softmaxClassiferGL1(X,y,10)

# Compute training and validation error
using Statistics
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),dims=2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);