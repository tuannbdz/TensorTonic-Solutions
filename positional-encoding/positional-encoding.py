import numpy as np
import math

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    
    position_encodings = np.zeros((seq_len, d_model + (d_model % 2)))

    position = np.arange(seq_len)[:, np.newaxis] # Shape (seq_len, 1)

    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(base) / d_model))
        
    position_encodings[:, 0::2] = np.sin(position * div_term)
    position_encodings[:, 1::2] = np.cos(position * div_term)

    if d_model % 2:
        position_encodings = position_encodings[:, :-1]
    return position_encodings

                