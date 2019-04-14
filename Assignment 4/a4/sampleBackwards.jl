using JLD
include("misc.jl")

function marginalCK(p1, pt, d)
    k = length(p1)
    M = zeros(d,k)
    M[1, :] = p1

    for i in 2:d
        for j in 1:k
            sum = 0
            for t in 1:k
                sum += pt[t, j]*M[i-1, t]
            end
            M[i, j] = sum
        end
    end

    return M
end

function sampleBackwards(p1, pt, d, xd)
    x = zeros(d)
    x[d] = xd
    mCK = marginalCK(p1, pt, d)
    xi_temp = xd
    for j in 0:d-2
        bckw_prob = pt[:, xi_temp] .* mCK[d - j - 1, :] ./ mCK[d - j, xi_temp]
        xi = sampleDiscrete(bckw_prob)
        x[d - j - 1] = xi
        xi_temp = xi

    end

    return x
end
