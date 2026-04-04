import numpy as np
def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here
    image = np.asarray(image)
    h, w = image.shape

    ys = np.linspace(0, h - 1, new_h)
    xs = np.linspace(0, w - 1, new_w)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    dy = (ys - y0)[:, None]
    dx = (xs - x0)[None, :]

    return (
        image[y0[:, None], x0[None, :]] * (1 - dy) * (1 - dx) +
        image[y1[:, None], x0[None, :]] * dy * (1 - dx) +
        image[y0[:, None], x1[None, :]] * (1 - dy) * dx +
        image[y1[:, None], x1[None, :]] * dy * dx
    ).tolist()
    