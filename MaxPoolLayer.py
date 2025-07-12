import numpy as np

class MaxPoolLayer:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        channels, H_in, W_in = input.shape
        k = self.kernel_size
        s = self.stride

        H_out = (H_in - k) // s + 1
        W_out = (W_in - k) // s + 1

        output = np.zeros((channels, H_out, W_out))
        self.max_indices = np.zeros_like(input, dtype=bool)

        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * s
                    w_start = j * s
                    window = input[c, h_start:h_start+k, w_start:w_start+k]
                    max_pos = np.unravel_index(np.argmax(window), window.shape)
                    output[c, i, j] = window[max_pos]
                    # Store mask for backward
                    self.max_indices[c, h_start + max_pos[0], w_start + max_pos[1]] = True

        return output

    def backward(self, dL_dout):
        """
        dL_dout: gradient of loss w.r.t. output of max pool, shape (channels, H_out, W_out)
        returns: gradient to pass to previous layer, same shape as input
        """
        dL_dinput = np.zeros_like(self.input)
        k = self.kernel_size
        s = self.stride
        channels, H_out, W_out = dL_dout.shape

        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * s
                    w_start = j * s
                    # Only one element gets the gradient — the one we stored as max
                    for m in range(k):
                        for n in range(k):
                            h = h_start + m
                            w = w_start + n
                            if self.max_indices[c, h, w]:
                                dL_dinput[c, h, w] = dL_dout[c, i, j]
        return dL_dinput
