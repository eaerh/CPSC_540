# Load X and y variable
using JLD, Statistics, Printf
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
"""
# Fit a least squares model
include("leastSquares.jl")
model = leastSquares(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
"""

# Fit a  absolute error model
include("leastSquares.jl")
model = leastAbsolutes(X,y)

# Evaluate training error

yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least absolute error: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least absolute error: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
savefig("leastAbsoluteError.pdf")
show()


# Fit a lbrittle regression model
include("leastSquares.jl")
model = leastMax(X,y)

# Evaluate training error
@show(size(y))
yhat = model.predict(X)
@show(size(yhat))
trainError = mean((yhat - y).^2)
@printf("Squared train Error with brittle regression: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with brittle regression: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
savefig("brittleRegression.pdf")
show()