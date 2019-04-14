include("logReg.jl")
include("misc.jl")

# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

# Reduce train set size for debugging
# X = X[:,:,1:100]

m = size(X,1)
n = size(X,3)

# Flatten X for easier handling
X_flat = zeros(n, m*m)
for i in 1:n
    X_flat[i, :] = reshape(X[:,:,i]', (m*m))
end

# Create array of Sigmoid probabilities
model = Array{SampleModel}(undef, m, m)
for i in 1:m
    for j in 1:m
        if !(i==1 && j==1)
            numPix = (i-1)*m + j
            X_bar = zeros(n, numPix-1)
            y_bar = X_flat[:, numPix]
            for k in 1:n
                X_bar[k, :] = deleteat!(X_flat[k, 1:numPix], numPix)
            end
            model[i, j] = logReg(X_bar, y_bar)
            print("Pixel nr: ", numPix, " ")
        end
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
    
    # Flatten I for easier handling
    I_flat = reshape(I', (m*m))

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            numPix = (i-1)*m + j
            if isnan(I_flat[numPix])
                x_tilde = deleteat!(I_flat[1:numPix], numPix)
                I[i,j] = model[i, j].sample(x_tilde)
                I_flat[numPix] = I[i,j]
            end
        end
    end
    imshow(I)
end