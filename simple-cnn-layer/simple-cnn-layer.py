import numpy as np
from numpy.lib.stride_tricks import as_strided

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape

    H_out = H - kH + 1
    W_out = W_in - kW + 1
    shape = (N, C_in, H_out, W_out, kH, kW)
    strides = (
        x.strides[0],          # N
        x.strides[1],          # C
        x.strides[2],          # move down (H)
        x.strides[3],          # move right (W)
        x.strides[2],          # kernel height
        x.strides[3],          # kernel width
    )

    x_patches = as_strided(x, shape=shape, strides=strides)
    out = np.einsum('ncijhw,ochw->noij', x_patches, W)
    out += b[None, :, None, None]

    return out
