import numpy as np
def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.array(image)
    kernel = np.array(kernel)
    H, W = image.shape
    k_h, k_w = kernel.shape
    o_h = (H + 2 * padding - k_h) // stride + 1
    o_w = (W + 2 * padding - k_w) // stride + 1
    p_img = np.pad(image, padding)
    out = np.zeros((o_h, o_w))
    for i in range(o_h):
        for j in range(o_w):
            out[i, j] = np.sum(p_img[i * stride : i * stride + k_h , \
                                     j * stride : j * stride + k_w] * kernel)
    return out.tolist()