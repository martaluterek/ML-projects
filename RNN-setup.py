"""
===============================================================================
Date        : 2025-11-12
Description : Functions implementing a Recurrent Neural Network (RNN) from scratch
using the tinygrad library. The RNN is designed to process sequences of characters
and learn to predict the next character in a sequence. The code includes an encoder
to convert characters to one-hot vectors, a step function to perform the RNN computations,
and a training loop to optimize the model parameters based on the cross-entropy loss.
===============================================================================
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tinygrad
from tinygrad.tensor import Tensor

# Encoder

def encoder(tokens):
    assert len(set(tokens)) == len(tokens), f'non-unique chars in tokens: {tokens}'
    map = {}
    rmap = {}
    for i, t in enumerate(tokens):
        map[t] = i
        rmap[i] = t

    def encode(data):
        result = np.zeros((len(tokens), len(data)))
        for i, d in enumerate(data):
            result[map[d], i] = 1
        return result

    def decode_one(data):
        assert isinstance(data, np.ndarray)
        for i, d in enumerate(data):
            if d != 0:
                return rmap[i]

    return encode, decode_one


class RNN():
    def __init__(self, input_size, hidden_size, tokens):
        self.batch_size = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encode, self.decode = encoder(tokens)
        self.h = Tensor(np.zeros((self.hidden_size)), dtype='float')
        self.W_xh = Tensor(np.random.uniform(-1, 1, (self.input_size, self.hidden_size)), dtype='float',
                           requires_grad=True)
        self.W_hh = Tensor(np.random.uniform(-1, 1, (self.hidden_size, self.hidden_size)), dtype='float',
                           requires_grad=True)
        self.W_hy = Tensor(np.random.uniform(-1, 1, (self.hidden_size, self.input_size)), dtype='float',
                           requires_grad=True)

    def __repr__(self):
        return f'h:\n{self.h.numpy()}\n----------\nxh:\n{self.W_xh.numpy()}\n----------\nhh:\n{self.W_hh.numpy()}\n----------\nhy:\n{self.W_hy.numpy()}'

    def step(self, input):
        input = Tensor(np.asarray(input), dtype='float')
        p1 = self.W_hh @ self.h.reshape(self.hidden_size, 1).flatten()
        p2 = (input.reshape(1, self.input_size) @ self.W_xh).flatten()
        self.h = (p1 + p2).tanh()
        y = self.h @ self.W_hy
        return y

    def forward(self, input):
        enc_input = self.encode(input)
        result = []
        for col in enc_input.T:
            result.append(self.step(col).numpy())

        return ''.join([self.decode(r) for r in result])

    def step_gradient(self, input, real_val):
        # forward pass
        input = Tensor(np.asarray(input), dtype='float')
        p1 = self.W_hh @ self.h.reshape(self.hidden_size, 1).flatten()
        p2 = (input.reshape(1, self.input_size) @ self.W_xh).flatten()
        self.h = (p1 + p2).tanh()
        y = self.h @ self.W_hy
        exp_y = np.exp(y)
        probs = exp_y / exp_y.sum()

        # backward pass
        real_val = self.encode([real_val])

        # compute the loss
        loss = -np.sum(real_val * np.log(probs))  # cross-entropy
        print('loss:', loss)
        # gradient
        d_y = probs - real_val
        print('d_y: ', d_y)

        return y

    def train(self, input, epochs=1000):
        pass

    def backward(self, input):
        pass

    def predict(self, input):
        pass


r = RNN(4,3,'helo')
r.step_gradient(r.encode('h'), 'e')
