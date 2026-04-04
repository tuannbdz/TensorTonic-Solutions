import numpy as np
def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here
    image = np.array(image)
    old_h, old_w = image.shape
    r_image = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            try:
                src_y = (old_h - 1) / (new_h - 1) * i
            except:
                src_y = 0
            try:
                src_x = (old_w - 1) / (new_w - 1) * j
            except:
                src_x = 0
            y0, x0 = int(np.floor(src_y)), int(np.floor(src_x))
            y1, x1 = min(y0 + 1, old_h - 1), min(x0 + 1, old_w - 1)
            dy, dx = src_y - y0, src_x - x0
            r_image[i, j] = image[y0, x0] * (1 - dy) * (1 - dx) + \
                            image[y1, x0] * dy * (1 - dx) + \
                            image[y0, x1] * (1 - dy) * dx + \
                            image[y1, x1] * dy * dx
    return list(r_image)
    