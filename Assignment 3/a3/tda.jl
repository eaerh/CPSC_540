include("misc.jl")
include("studentT.jl")


function tda(X, y, Xtest)

    k = maximum(y)
    (n,d) = size(X)
    a = countmap(y)
    theta = zeros(k)

    subModel = Array{DensityModel}(undef,k)

    for c in 1:k
        theta[c] = a[c]/n

        x = X[findall(x -> x == c, y ), :]


        subModel[c] = studentT(x)

    end

    function predict(X)
        (t, d) = size(X)
        yhat = zeros(t)

        pdfs = zeros(t,k)

        for c in 1:k

            pdfs[:,c] = theta[c] * subModel[c].pdf(X)
        end

        for i in 1:t
            yhat[i] = findmax(pdfs[i,:])[2]
        end

        return yhat
    end

    yhat = predict(Xtest)
    return yhat

end
