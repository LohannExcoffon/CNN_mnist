import numpy as np
import matplotlib.pyplot as plt

class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate=0.01, padding=0, w=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.learning_rate = learning_rate
        if w != 0:
            self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
            self.weights[0][0] = [[1,1,1],[0,0,0],[-1,-1,-1]]
            self.weights[1][0] = [[1,0,-1],[1,0,-1],[1,0,-1]]
            self.weights[2][0] = [[-1,0,1],[-1,0,1],[-1,0,1]]
            self.weights[3][0] = [[-1,-1,-1],[0,0,0],[1,1,1]]
        else:
            self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1

        self.biases = np.zeros(out_channels)

    def conv2d_single_channel(self, input, kernel):
        new_input = np.zeros((input.shape[0] + self.padding * 2, input.shape[1] + self.padding * 2))
        H_in, W_in = new_input.shape

        if self.padding <= 0:
            new_input = input.copy()
        else:
            new_input[self.padding:H_in-self.padding, self.padding:W_in-self.padding] = input

        kH, kW = kernel.shape
        H_out = H_in - kH + 1
        W_out = W_in - kW + 1
        output = np.zeros((H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                output[i, j] = np.sum(new_input[i:i+kH, j:j+kW] * kernel)
        plt.show()
        return output

    def forward(self, input):
        self.input = input
        in_channels, H_in, W_in = input.shape
        k = self.kernel_size
        H_out = H_in - k + 1 + 2*self.padding
        W_out = W_in - k + 1 + 2*self.padding

        output = np.zeros((self.out_channels, H_out, W_out))
        self.output = output

        for out_c in range(self.out_channels):
            sum_conv = np.zeros((H_out, W_out))
            for in_c in range(in_channels):
                conv = self.conv2d_single_channel(input[in_c], self.weights[out_c, in_c])
                sum_conv += conv
            sum_conv += self.biases[out_c]
            output[out_c] = np.maximum(0, sum_conv)  # ReLU
        return output

    def backward(self, dL_dout):
        in_channels, H_in, W_in = self.input.shape
        k = self.kernel_size
        _, H_out, W_out = dL_dout.shape
        pad = self.padding
    
        # Pad input for window extraction
        if pad > 0:
            input_padded = np.pad(self.input, ((0,0), (pad,pad), (pad,pad)), mode='constant')
        else:
            input_padded = self.input
    
        dL_dinput_padded = np.zeros_like(input_padded)
    
        self.dL_dweights = np.zeros_like(self.weights)
        self.dL_dbiases = np.zeros_like(self.biases)
    
        relu_mask = self.output > 0
        dL_dout = dL_dout * relu_mask
    
        for out_c in range(self.out_channels):
            for in_c in range(self.in_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i
                        w_start = j
                        # Use padded input here
                        window = input_padded[in_c, h_start:h_start+k, w_start:w_start+k]
    
                        self.dL_dweights[out_c, in_c] += dL_dout[out_c, i, j] * window
                        dL_dinput_padded[in_c, h_start:h_start+k, w_start:w_start+k] += dL_dout[out_c, i, j] * self.weights[out_c, in_c]
            self.dL_dbiases[out_c] += np.sum(dL_dout[out_c])
    
        # Remove padding from dL_dinput before returning
        if pad > 0:
            dL_dinput = dL_dinput_padded[:, pad:-pad, pad:-pad]
        else:
            dL_dinput = dL_dinput_padded
    
        return dL_dinput


    def update(self):
        self.weights -= self.learning_rate * self.dL_dweights
        self.biases -= self.learning_rate * self.dL_dbiases
