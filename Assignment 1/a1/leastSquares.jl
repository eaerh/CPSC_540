include("misc.jl")

using MathProgBase, GLPKMathProgInterface

function leastSquares(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

using LinearAlgebra

function leastAbsolutes(X, y)

	(n, d) = size(X)


	b = zeros(2*n)
	a = zeros(2*n)
	B = zeros(2*n)
	R = zeros(2*n,n)

	counter = 1
	for i in 1:2*n
		if (i % 2 == 0)
			b[i] = y[counter]
			a[i] = X[counter]
			R[i, counter] = -1
			B[i] = 1
			counter = counter + 1
		else
			b[i] = -y[counter]
			a[i] = -X[counter]
			R[i, counter] = -1
			B[i] = -1
		end
	end


	A = [a B R ]

	zeroVec = zeros(1,2 )
	oneVec = ones(1,n)

	f = [zeroVec oneVec]

	f = reshape(f, length(f),1)
	f = vec(f)
	ub = Inf*ones(502)
	lb = zeros(502)
	lb[1] = -Inf
	d = -Inf*ones(1000)


	solution = linprog(f,A,d,b,lb,ub,GLPKSolverLP())
	w = solution.sol

	predict(Xtilde) = [Xtilde ones(size(Xtilde,1),1) ]*w[1:2]

	return LinearModel(predict,w)
end

function leastMax(X,y)
	(n, d) = size(X)


	b = zeros(2*n)
	a = zeros(2*n)
	B = zeros(2*n)
	R = ones(2*n,n)

	counter = 1
	for i in 1:2*n
		if (i % 2 == 0)
			b[i] = y[counter]
			a[i] = X[counter]
			#R[i, counter] = -1
			B[i] = 1
			counter = counter + 1
		else
			b[i] = -y[counter]
			a[i] = -X[counter]
			#R[i, counter] = -1
			B[i] = -1
		end
	end

	r = -1*ones(2*n)

	A = [a B r ]

	zeroVec = zeros(1,2 )
	oneVec = ones(1,1)

	f = [zeroVec oneVec]

	f = reshape(f, length(f),1)
	f = vec(f)
	ub = Inf*ones(length(f))
	lb = zeros(length(f))
	lb[1] = -Inf
	d = -Inf*ones(2*n)


	solution = linprog(f,A,d,b,lb,ub,GLPKSolverLP())
	w = solution.sol

	predict(Xtilde) = [Xtilde ones(size(Xtilde,1),1) ]*w[1:2]

	return LinearModel(predict,w)
end
