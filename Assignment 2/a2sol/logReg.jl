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

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

# Multi-class softmax version (assumes y_i in {1,2,...,k})
function logRegSoftmax(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y,k)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll,reshape(G,d*k,1))
end

function logRegSoftmaxL2(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObjL2(w,X,y,k,lambda)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end




function softmaxObjL2(w,X,y,k,lambda)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
    
    # Add L2 regularization to objective function
    nll += lambda*sum(W.*W)
    
    # Add L2 regularization to objective gradient
    G += 2*lambda*W
    
	return (nll,reshape(G,d*k,1))
end




function logRegSoftmaxL1(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObj(w,X,y,k)

	W[:] = findMinL1(funObj,W[:],lambda,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)
    
	return LinearModel(predict,W)
end




function logRegSoftmaxGL1(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	funObj(w) = softmaxObjL2(w,X,y,k,lambda)

	W[:] = findMin(funObj,W[:],derivativeCheck=true,maxIter=500)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

    

function proxGradGroupL1(funObj,w,d,lambda;maxIter=100)
    
    alpha = 1
    gamma = 1e-4
    (f,g) = funObj(w)
    
    for i in 1:maxIter
        # Gradient step on smoooth part
        wNew = w - alpha*g
        
        # Groupwise proximal step on non-smooth part
        for i in 1:d
            wNew[i:d:end] = wNew[i:d:end]*max(0,norm(wNew[i:d:end],2)-alpha*lambda)/norm(wNew[i:d:end],2)
        end
        
        # Setting new objective values
        (fNew,gNew) = funObj(wNew)
        
        # Reduce step size if function value increases
        while fNew > f
            alpha /= 2
            # Gradient step on smoooth part
            wNew = w - alpha*g
        
            # Groupwise proximal step on non-smooth part
            for i in 1:d
                wNew[i:d:end] = wNew[i:d:end]*max(0,norm(wNew[i:d:end],2)-alpha*lambda)/norm(wNew[i:d:end],2)
            end
            
            # Setting new objective values
            (fNew,gNew) = funObj(wNew)
        end
        
        # Accept new parameters
        w = wNew
        f = fNew
        g = gNew
    end
    
    return w
end





function softmaxClassiferGL1(X,y,lambda)
    (n,d) = size(X)
    k = maximum(y)
    
    # Each column of 'w' will be a logistic regression classifier
    W = zeros(d,k)
        
    funObj(w) = softmaxObj(w,X,y,k)
        
    W[:] = proxGradGroupL1(funObj,W[:],d,lambda,maxIter=500)

    # Make linear prediction function
    predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)
    
    return LinearModel(predict,W)

end
