import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from .meshgen import meshgen

def mu2map(mu_pad0):
    """
    Inputs:
        mu_pad0 : (h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        mapping: (2, h, w)
        Ax, Ay: (h*w, h*w)
        bx, by: (h*w, 1)
    """
    # a = time.time()
    h, w = mu_pad0.shape
    hw = h*w
    Ax = mu2A(mu_pad0).tolil()
    Ay = Ax.copy()
    bx = np.zeros((hw, 1))
    by = bx.copy()

    Edge1 = np.reshape(np.arange((h-1)*w, hw), (w, 1))
    Edge2 = np.reshape(np.arange(w-1, hw, step = w), (h, 1))
    Edge3 = np.reshape(np.arange(w), (w, 1))
    Edge4 = np.reshape(np.arange((h-1)*w+1, step = w), (h, 1))

    landmarkx = np.vstack((Edge4, Edge2))
    targetx = np.vstack((np.zeros_like(Edge4), np.ones_like(Edge2)))
    lmx = landmarkx.reshape(-1)
    bx[lmx] = targetx
    Ax[lmx, :] = 0
    tmp = sps.csr_matrix((np.ones_like(lmx), (np.arange(lmx.shape[0]), lmx)), shape = (lmx.shape[0], hw)).tolil()
    Ax[lmx, :] = tmp
    mapx = spsolve(Ax.tocsc(), bx).reshape((h, w))#.reshape((-1, 1))

    landmarky = np.vstack((Edge1, Edge3))
    targety = np.vstack((np.zeros_like(Edge1), np.ones_like(Edge3)))
    lmy = landmarky.reshape(-1)
    by[lmy] = targety
    Ay[lmy, :] = 0
    tmp = sps.csr_matrix((np.ones_like(lmy), (np.arange(lmy.shape[0]), lmy)), shape = (lmy.shape[0], hw)).tolil()
    Ay[lmy, :] = tmp
    mapy = spsolve(Ay.tocsc(), by).reshape((h, w))#.reshape((-1, 1))

    mapping = np.array((mapx, mapy))
    return mapping, Ax, Ay, bx, by

def mu2A(mu_pad0):
    """
    Inputs:
        mu_pad0 : (h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
    """
    # a = time.time()
    h, w = mu_pad0.shape
    mu = mu_pad0[:h-1, :w-1]

    mu_reshape = np.zeros(((h-1), (w-1)*2), dtype = np.complex)
    mu_reshape[:, ::2] = mu
    mu_reshape[:-1, 1:-1:2] = (mu[:-1, :-1] + mu[:-1, 1:] + mu[1:, :-1]) / 3
    mu_reshape[:-1, -1] = (mu[:-1, -1] + mu[1:, -1]) / 2
    mu_reshape[-1, 1:-1:2] = (mu[-1, :-1] + mu[-1, 1:]) / 2
    mu_reshape[-1, -1] = mu[-1, -1]
    mu = mu_reshape.reshape((-1, 1))

    A = lbs(mu, h, w);
    #x = mapping[:, 0].reshape((h, w))[np.newaxis, ...]
    #y = mapping[:, 1].reshape((h, w))[np.newaxis, ...]
    #mapping = np.vstack((x, y))
    # b = time.time()
    # print('Time(s) of solving Ax=b:', b-a)
    #return mapping
    return A

def lbs(mu, h, w):
    """
    Inputs:
        mu : m x 1 Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
    """
    face, vertex = meshgen(h, w)
    A, abc, area = generalized_laplacian2D(face, vertex, mu, h, w)
    return A

def generalized_laplacian2D(face, vertex, mu, h, w):
    """
    Inputs:
        face : m x 3 index of triangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
        mu : m x 1 Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
        abc : vectors containing the coefficients alpha, beta and gamma (m, 3)
        area : float, area of every triangles in the mesh
    """
    af = (1 - 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    bf = -2 * np.imag(mu) / (1 - np.abs(mu)**2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    abc = np.hstack((af, bf, gf))

    f0, f1, f2 = face[:, 0, np.newaxis], face[:, 1, np.newaxis], face[:, 2, np.newaxis]

    uxv0 = vertex[f1,1] - vertex[f2,1]
    uyv0 = vertex[f2,0] - vertex[f1,0]
    uxv1 = vertex[f2,1] - vertex[f0,1]
    uyv1 = vertex[f0,0] - vertex[f2,0]
    uxv2 = vertex[f0,1] - vertex[f1,1]
    uyv2 = vertex[f1,0] - vertex[f0,0]

    area = (1/(h-1)) * (1/(w-1)) / 2

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area;
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area;
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area;

    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area;
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area;
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area;

    I = np.vstack((f0,f1,f2,f0,f1,f1,f2,f2,f0)).reshape(-1)
    J = np.vstack((f0,f1,f2,f1,f0,f2,f1,f0,f2)).reshape(-1)
    nRow = vertex.shape[0]
    V = np.vstack((v00,v11,v22,v01,v01,v12,v12,v20,v20)).reshape(-1) / 2
    A = sps.coo_matrix((-V, (I, J)), shape = (nRow, nRow))

    return A, abc, area
