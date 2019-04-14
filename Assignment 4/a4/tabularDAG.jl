include("tabular.jl")
include("misc.jl")

# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

# Reduce train set size for debugging
#X = X[:,:,1:100]

m = size(X,1)
n = size(X,3)

# Create array of tabular DAG probabilities
model = Array{SampleModel}(undef, m, m)
for i in 3:m
    for j in 3:m
        X_bar = zeros(n, 8)
        y_bar = X[i, j, :]
        for k in 1:n
            X_bar[k, :] = deleteat!(reshape(X[i-2:i, j-2:j, k], 9), 9)
        end
        model[i, j] = tabular(X_bar, y_bar)
    end
end

# Fill-in some random test images
t = size(Xtest,3)
figure(2)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]
    
    # Set outer left edge to 0
    I[:, 1:2] = zeros(m, 2)

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                x_tilde = deleteat!(reshape(I[i-2:i, j-2:j], 9), 9)
                I[i,j] = model[i,j].sample(x_tilde)
            end
        end
    end
    imshow(I)
end
