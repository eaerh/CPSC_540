include("misc.jl")

using LinearAlgebra
function leastSquaresRBFL2(X, y, lambda, sigma)

	# Add bias column + RBF
	(n, d) = size(X)


	Xdist = distancesSquared(X, X)
	G = rbfBasis(lambda, sigma, Xdist)


	Z = [ones(n,1) X G]


	I = UniformScaling(1)

	# Find regression weights minimizing squared error
	v = (Z'*Z + lambda*I)\(Z'*y)

	# Make linear prediction function
	predict(Ztilde) = [ones(size(Ztilde,1),1) Ztilde ] * v

	return LinearModel(predict, v)
end


function rbfBasis(lambda, sigma, X)
	(n, d) = size(X)
	G = zeros(n, d)

	
	for i in 1:n
		for j in 1:d
			G[i, j] = exp(- X[i,j]^2/(2 * sigma^2))
		end
	end

	return G
end
