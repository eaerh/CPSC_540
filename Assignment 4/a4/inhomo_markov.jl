# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

# Reduce train set size for debugging
#X = X[:,:,1]

m = size(X,1)
n = size(X,3)

# Train inhomogenious Markov model
p_ji = zeros(m,m,2)
p_ji[1,:,:] .= sum(X[1,:,:] .== 1)/n
for i in 1:m
    for j in 2:m
        p_ji[j,i,1] = sum(X[j,i,X[j-1,i,:] .== 1] .== 1)/n
        p_ji[j,i,2] = sum(X[j,i,X[j-1,i,:] .== 0] .== 1)/n
    end
end

show(maximum(p_ji[:,:,1]))
show(maximum(p_ji[:,:,2]))
#show(p_ji[:,:,2])

# Show Markov parameters
figure(1)
imshow(p_ji[:,:,1])
imshow(p_ji[:,:,2])

# Fill-in some random test images
t = size(Xtest,3)
figure(2)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[j,i])
                if I[j-1,i] == 1
                    I[j,i] = rand() < p_ji[j,i,1]
                end
                if I[j-1,i] == 0
                    I[j,i] = rand() < p_ji[j,i,2]
                end
            end
        end
    end
    imshow(I)
end
