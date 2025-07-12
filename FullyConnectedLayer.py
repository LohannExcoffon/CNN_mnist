import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros(output_size)
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = input  # Save for backprop
        self.output = np.dot(self.weights, input) + self.biases
        return self.output

    def backward(self, dL_dz):
        """
        dL_dz: gradient of loss w.r.t. output logits (shape: (output_size,))
        Returns gradient to pass to previous layer: dL/dx (shape: (input_size,))
        """
        # Gradients w.r.t. parameters
        self.dL_dW = np.outer(dL_dz, self.input)  # (output, input)
        self.dL_db = dL_dz  # shape (output,)
        # Gradient w.r.t. input to this layer
        dL_dx = np.dot(self.weights.T, dL_dz)  # shape (input_size,)
        return dL_dx

    def update(self):
        # Gradient descent update
        self.weights -= self.learning_rate * self.dL_dW
        self.biases -= self.learning_rate * self.dL_db