from MaxPoolLayer import MaxPoolLayer
from FullyConnectedLayer import FullyConnectedLayer
from ConvolutionLayer import ConvolutionLayer

import numpy as np

# Define your CNN class wrapping everything
class SimpleCNN:
    def __init__(self):
        self.conv1 = ConvolutionLayer(in_channels=1, out_channels=4, kernel_size=3, learning_rate=0.01, padding=1, w=1)
        self.pool1 = MaxPoolLayer(kernel_size=3, stride=1)
        # self.conv2 = ConvolutionLayer(in_channels=4, out_channels=2, kernel_size=2, learning_rate=0.01)
        # self.pool2 = MaxPoolLayer(kernel_size=3, stride=1)
        self.fc = FullyConnectedLayer(input_size=4*26*26, output_size=10, learning_rate=0.01)

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        return exps / np.sum(exps)
    
    def cross_entropy_loss(self, probs, true_label):
        # Small epsilon to avoid log(0)
        epsilon = 1e-12
        return -np.log(probs[true_label] + epsilon)
    
    def forward(self, x):
        self.x1 = self.conv1.forward(x)
        self.x2 = self.pool1.forward(self.x1)
        # self.x3 = self.conv2.forward(self.x2)
        # self.x4 = self.pool2.forward(self.x3)

        self.x_flat = self.x2.flatten()
        self.logits = self.fc.forward(self.x_flat)
        self.probs = self.softmax(self.logits)
        return self.probs

    def backward(self, label):
        dL_dz = self.probs.copy()
        dL_dz[label] -= 1  # softmax + cross-entropy
        dL_flat = self.fc.backward(dL_dz)
        self.fc.update()

        # dL_pool2 = dL_flat.reshape(self.x4.shape)
        # dL_conv2 = self.pool2.backward(dL_pool2)
        # self.conv2.backward(dL_conv2)
        # self.conv2.update()
        dL_pool1 = dL_flat.reshape(self.x2.shape)
        #dL_pool1 = self.conv2.backward(dL_conv2) 

        dL_pool1 = self.pool1.backward(dL_pool1)
        self.conv1.backward(dL_pool1)
        self.conv1.update()

    def train_step(self, x, label):
        probs = self.forward(x)
        loss = self.cross_entropy_loss(probs, label)
        self.backward(label)
        return loss

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)
