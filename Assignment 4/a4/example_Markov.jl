# Load X and y variable
using JLD
include("misc.jl")
include("sampleAncestral.jl")
include("sampleBackwards.jl")
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)


function MCMC(p1, pt)
    N = 10000
    d = 50
    n = length(p1)
    marginal = zeros(n)

    conditional = zeros(n)

    conditional_fixedx1 = zeros(n)

    marginal_x10 = 0

    conditional_bckw = zeros(n)

    for i in 1:N
        (x_cond, prob_cond) = sampleAncestral(p1, pt, d, true)
        (x, prob) = sampleAncestral(p1, pt, d, false)

        #Code for Q1.2.2
        x_bckw = sampleBackwards(p1, pt, 10, 6)

        marginal[Int(x[d])] += 1
        conditional_fixedx1[Int(x_cond[d])] += 1
        conditional_bckw[Int(x_bckw[5])] += 1
        #Code for Q1.2.1
        if (x[10] == 6)
            marginal_x10 += 1
        end
        for j in 1:n
            if (x[10] == 6 && x[5] == j)
                conditional[j] += 1
            end
        end
    end
    conditional /= marginal_x10
    marginal /= N
    conditional_fixedx1 /= N
    conditional_bckw /= N

    return marginal, conditional, marginal_x10, conditional_fixedx1, conditional_bckw
end

(marginals, conditionals, x10, fixed_x0, bckw) = MCMC(p1, pt)
println("The marginal probabilities: ", marginals)
println("The conditional probabilities P(x5 = c | x10 = 6): ", conditionals)
println("Number of samples accepted: ", x10)
println("The conditional probabilities P(x50 = c | x0 = 3): ", fixed_x0)
println("The conditional probabilities P(x5 = c | x10 = 6), using backwards sampling: ", bckw)
