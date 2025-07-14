from PIL import Image
import pandas as pd
import numpy as np
import os
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from SimpleCNN import SimpleCNN
from MaxPoolLayer import MaxPoolLayer
from FullyConnectedLayer import FullyConnectedLayer
from ConvolutionLayer import ConvolutionLayer


def load_images(folder, label, image_size=(28, 28)):
    data = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')  # grayscale
        img = img.resize(image_size)
        img_array = np.asarray(img).flatten() / 255.0
        data.append((img_array, label))
    return data

def convolve(input, kernel):
    """
    input: numpy array of shape (H_in, W_in)
    kernel: numpy array of shape (kH, kW)
    returns: output of convolution (H_out, W_out)
    """
    H_in, W_in = input.shape
    kH, kW = kernel.shape
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            # Dot Product
            output[i, j] = np.sum(input[i:i+kH, j:j+kW] * kernel)

    return output



# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values and make grayscale
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Reshape to 2D
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
print(x_train.shape)  # (60000, 1, 28, 28)
print(x_test.shape)   # (10000, 1, 28, 28)

# INITIALIZE MODEL
cnn = SimpleCNN()

epochs = 10
pool = 60000
num_train = 500
indices = np.random.choice(pool, size=num_train*epochs, replace=False)
train_images = x_train[indices]
train_labels = y_train[indices]

# === TRAINING LOOP ===
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    # Shuffle training data
    train_images1 = train_images[:num_train*(epoch+1)]
    train_labels1 = train_labels[:num_train*(epoch+1)]

    for i in range(num_train):
        X = train_images1[i]  # shape (1, 28, 28)
        y = train_labels1[i]

        loss = cnn.train_step(X, y)
        total_loss += loss

        # Track accuracy
        pred = cnn.predict(X)
        if pred == y:
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{num_train}] Loss: {total_loss / num_train:.4f}")

    print(f"Epoch {epoch + 1}/{epochs} — Avg Loss: {total_loss / num_train:.4f}, Accuracy: {correct / num_train:.4f}")


correct = 0
wrong = []
num = 1000
for i in range(num):
    x = x_test[i]  # shape (1, 28, 28)
    y = y_test[i]
    pred = cnn.predict(x)
    if pred == y:
        correct += 1
    else:
        wrong.append(x_test[i])

accuracy = (correct / num ) * 100
print(f"CNN got {accuracy}% right on test set")

test = []
for i in range(len(y_train)):
    if y_train[i] == 2 or y_train[i] == 9:
        test.append(i)


def visualize_filters(conv_layer):
    filters = conv_layer.weights  # shape: (out_channels, in_channels, kH, kW)
    num_filters = filters.shape[0]

    fig, axs = plt.subplots(1, num_filters, figsize=(num_filters * 2.5, 2.5))
    if num_filters == 1:
        axs = [axs]

    for i in range(num_filters):
        kernel = filters[i, 0]  # Grab the 2D filter for channel 0
        axs[i].imshow(kernel, cmap='gray')
        axs[i].set_title(f"Filter {i + 1}")
        axs[i].axis('off')

    plt.suptitle("Learned Convolutional Filters")
    plt.show()

# Call it after training
visualize_filters(cnn.conv1)
#visualize_filters(cnn.conv2)

def visualize_feature_maps(model, image):
    # Forward through conv layer only
    conv_output = model.conv1.forward(image)  # shape: (out_channels, H, W)
    num_filters = conv_output.shape[0]

    fig, axs = plt.subplots(1, num_filters, figsize=(num_filters * 3, 3))
    if num_filters == 1:
        axs = [axs]

    for i in range(num_filters):
        axs[i].imshow(conv_output[i], cmap='gray')
        axs[i].set_title(f"Feature Map {i + 1}")
        axs[i].axis('off')

    plt.suptitle("Convolutional Feature Maps")
    plt.show()

    # Print prediction
    probs = model.forward(image)
    pred = np.argmax(probs)
    print(f"Predicted Digit: {pred}")
    print("Class Probabilities:", np.round(probs, 3))

    #  # Forward through conv layer only
    # conv_output = model.conv2.forward(image)  # shape: (out_channels, H, W)
    # num_filters = conv_output.shape[0]

    # fig, axs = plt.subplots(1, num_filters, figsize=(num_filters * 3, 3))
    # if num_filters == 1:
    #     axs = [axs]

    # for i in range(num_filters):
    #     axs[i].imshow(conv_output[i], cmap='gray')
    #     axs[i].set_title(f"Feature Map {i + 1}")
    #     axs[i].axis('off')

    # plt.suptitle("Convolutional Feature Maps")
    # plt.show()

    # # Print prediction
    # probs = model.forward(image)
    # pred = np.argmax(probs)
    # print(f"Predicted Digit: {pred}")
    # print("Class Probabilities:", np.round(probs, 3))

print(x_train[0].shape)
for i in range(3):
    visualize_feature_maps(cnn, x_test[i+100])

for i in range(5):
    visualize_feature_maps(cnn,wrong[i])


# # Forward
# x = cnn.conv.forward(img)                       # (3, 26, 26)
# x = cnn.pool.forward(x)                         # (3, 13, 13)
# x_flat = x.flatten()
# logits = cnn.fc.forward(x_flat)
# probs = softmax(logits)
# loss = cross_entropy_loss(probs, label)

# # Backward
# dL_dz = probs.copy()
# dL_dz[label] -= 1
# dL_dx_flat = cnn.fc.backward(dL_dz)
# cnn.fc.update()

# dL_dpool = dL_dx_flat.reshape(3, 13, 13)
# dL_dconv = cnn.pool.backward(dL_dpool)
# dL_dinput = cnn.conv.backward(dL_dconv)
# cnn.conv.update()


# output_probs = cnn.forward(sample_img)

# print("Output probabilities:", output_probs)
# print("Sum (should be 1):", np.sum(output_probs))
# print("Predicted class:", np.argmax(output_probs))
# print("True label:", y_train[0])


# # Load dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize pixel values to the range [0, 1]
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# # Get images in 2D shape
# x_train = x_train.reshape(-1, 1, 28, 28)
# x_test = x_test.reshape(-1, 1, 28, 28)

# print(x_train.shape)  # (60000, 1, 28, 28)
# print(x_test.shape)   # (10000, 1, 28, 28)



# # Example kernel: simple edge detector
# kernel = np.array([
#     [-1, 0, 1],
#     [-1, 0, 1],
#     [-1, 0, 1]
# ])

# # Pick an image (28x28)
# img = x_train[0, 0, :, :]  # shape (28,28)

# conv_result = convolve(img, kernel)

# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(img, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title("Convolved")
# plt.imshow(conv_result, cmap='gray')

# plt.show()



# conv_layer = ConvolutionLayer(in_channels=1, out_channels=3, kernel_size=3)

# # Pick a sample MNIST image, shape (1, 28, 28)
# sample_img = x_train[0]

# # Forward pass through conv layer
# output = conv_layer.forward(sample_img)

# print("Output shape:", output.shape)  # Should be (3, 26, 26)

# fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# for i in range(3):
#     axs[i].imshow(output[i], cmap='gray')
#     axs[i].set_title(f'Filter {i+1}')
#     axs[i].axis('off')
# plt.show()



# pool = MaxPoolLayer(kernel_size=2, stride=2)
# pooled_output = pool.forward(output)

# print("Pooled output shape:", pooled_output.shape)  # Should be (3, 13, 13)

# fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# for i in range(3):
#     axs[i].imshow(pooled_output[i], cmap='gray')
#     axs[i].set_title(f'Pooled Filter {i+1}')
#     axs[i].axis('off')
# plt.show()



# fc = FullyConnectedLayer(input_size=507, output_size=10)

# logits = fc.forward(pooled_output.flatten())
# print("Logits shape:", logits.shape) 
# print("Logits:", logits)

# probs = softmax(logits)
# print("Probabilities:", probs)
# print("Sum:", np.sum(probs))



# def conv2d_single_channel(input, kernel, padding=0):
#     new_input = np.zeros((input.shape[0]+padding*2, input.shape[1]+padding*2))
#     H_in, W_in = new_input.shape

#     if padding <= 0:
#         new_input = input.copy()
#     else:
#         new_input[padding:H_in-padding, padding:W_in-padding] = input

#     kH, kW = kernel.shape
#     H_out = H_in - kH + 1
#     W_out = W_in - kW + 1
#     output = np.zeros((H_out, W_out))

#     for i in range(H_out):
#         for j in range(W_out):
#             output[i, j] = np.sum(new_input[i:i+kH, j:j+kW] * kernel)
#     plt.show()

# input = np.random.randint(1, 11, size=(5,5))
# print(input)
# kernel = np.random.randn(3,3)
# conv2d_single_channel(input,kernel)
    