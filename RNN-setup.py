"""
===============================================================================
Author      : Marta Luterek
Date        : 2025-11-12
Description : Functions implementing a Recurrent Neural Network (RNN)
===============================================================================
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tinygrad
from tinygrad.tensor import Tensor

help(assert)

# Encoder
def encoder(tokens):
    assert len(set(tokens)) == len(tokens),