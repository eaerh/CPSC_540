using JLD, Printf, Statistics, LinearAlgebra
include("misc.jl")


function gda(X, y, Xtest)

    k = maximum(y)
    (n,d) = size(X)
    a = countmap(y)
    theta = zeros(k)
    mean_ = zeros(k, d)
    covar = zeros(d, d, k)



    for c in 1:k
        theta[c] = a[c]/n
        mean_[c,:] = 1/a[c] * sum(X[findall(x -> x == c, y), :], dims=1)
        A = zeros(d,d)
        x = X[findall(x -> x == c, y ), :]
        for j in 1:a[c]
            A = A + (x[j,:] - mean_[c,:]) * (x[j,:] - mean_[c,:])'
        end
        covar[:,:,c] = 1/a[c] * A
    end

    (n, d) = size(Xtest)
    yhat = zeros(n)


    for i in 1:n
        probvec = zeros(k)
        for c in 1:k
            probvec[c] = log(theta[c]) - 1/2 * logdet(covar[:,:,c]) - 1/2 * (Xtest[i,:] - mean_[c,:])' * inv(covar[:,:,c]) * (Xtest[i,:] - mean_[c,:])
        end
        yhat[i] = findmax(probvec)[2]

    end

    return yhat

end
