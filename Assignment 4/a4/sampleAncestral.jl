include("misc.jl")

function sampleAncestral(p1, pt, d, fixed)

    x = zeros(d)

    #Code for Q1.1.4
    if (fixed == true)
        x0 = 3
        xi_temp = x0
        prob = p1[x0]
        x[1] = x0
    else
        x0 = sampleDiscrete(p1)
        prob = p1[x0]
        xi_temp = x0
        x[1] = x0
    end

    for i in 2:d

        xi = sampleDiscrete(pt[xi_temp,:])
        x[i] = xi
        prob = prob * pt[xi_temp, xi]
        xi_temp = xi
    end
    return x, prob
end
