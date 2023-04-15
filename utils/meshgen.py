import numpy as np

def meshgen(height, width, return_boundary_mask = False):
    """
    Inputs:
        height: int
        width: int
    Outputs:
        face : m x 3 index of trangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)

    Notes:
        if return_boundary_mask is True, then return a mask in which the faces adjacent to boundary vertices is 1 while the other faces is 0.
    """    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    y = y[::-1, :]
    x, y = x/(width-1), y/(height-1)
    x, y = x.reshape((-1, 1)), y.reshape((-1, 1))
    vertex = np.hstack((x, y))
    face = np.zeros(((height-1)*(width-1)*2, 3)).astype(np.int)

    # without for loop
    ind = np.arange(height*width).reshape((height, width))
    mid = ind[0:-1, 1:]
    left1 = ind[0:-1, 0:-1]
    left2 = ind[1:, 1:]
    right = ind[1:, 0:-1]
    face[0::2, 0] = left1.reshape(-1)
    face[0::2, 1] = right.reshape(-1)
    face[0::2, 2] = mid.reshape(-1)
    face[1::2, 0] = left2.reshape(-1)
    face[1::2, 1] = mid.reshape(-1)
    face[1::2, 2] = right.reshape(-1)

    if return_boundary_mask:
        # boundary is 1, the other faces is 0
        boundary_mask = np.zeros_like(face).astype(np.int)
        ind_mask = np.arange(height*width).reshape((height, width)) + 1
        ind_mask[1:-1, 1:-1] = 0
        ind_mask[ind_mask > 0] = 1
        mid_mask = ind_mask[0:-1, 1:]
        left1_mask = ind_mask[0:-1, 0:-1]
        left2_mask = ind_mask[1:, 1:]
        right_mask = ind_mask[1:, 0:-1]
        boundary_mask[0::2, 0] = left1_mask.reshape(-1)
        boundary_mask[0::2, 1] = right_mask.reshape(-1)
        boundary_mask[0::2, 2] = mid_mask.reshape(-1)
        boundary_mask[1::2, 0] = left2_mask.reshape(-1)
        boundary_mask[1::2, 1] = mid_mask.reshape(-1)
        boundary_mask[1::2, 2] = right_mask.reshape(-1)
        boundary_mask = (np.sum(boundary_mask, axis = 1) > 0) + 0.0

        return face, vertex, boundary_mask

    return face, vertex

