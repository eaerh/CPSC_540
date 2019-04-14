include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=true)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxClassifier(X, y)
	(n, d) = size(X)
	k = maximum(y)
	w = zeros(d*k)


	funObj(w) = softmaxObj(w, X, y)
	W = findMin(funObj, w, derivativeCheck=true,verbose=true)


	W = reshape(W, d, k)


	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)
	return LinearModel(predict, W)
end

function softmaxObj(w, X, y)
	(n, d) = size(X)
	k = maximum(y)

	w = reshape(w, (d, k))
	f = 0
	g = zeros(d, k)

	#classSum = zeros(n)
	classSum = sum(exp.(X*w),dims=2)
	for i in 1:n

		f += -(w[:,y[i]]'*X[i,:]) + log.(classSum[i])
	end

	for c in 1:k
		for j in 1:d
			for i in 1:n
				g[j, c] += X[i,j] * (exp.(w[:,c]'*X[i,:])/classSum[i] - (y[i]==c))
			end

		end
	end

	g = reshape(g, (d*k))
	return (f,g)
end


