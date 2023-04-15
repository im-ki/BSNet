import torch
from .meshgen import meshgen

def batch_mu2reshape_torch(mu_pad0, device):
    """
    Inputs:
        mu_pad0 : (N, 2, h, w), the elements in the right-most column and bottom row are padded by 0.
    Outputs:
        mu : (N, 2, (h-1)*(w-1)*2), add removed elements back to the input mu_pad0
    """
    N, _, h, w = mu_pad0.shape
    mu = mu_pad0[:, :, :h-1, :w-1]

    mu_reshape = torch.zeros((N, 2, (h-1), (w-1)*2), device = device)
    mu_reshape[:, :, :, ::2] = mu
    mu_reshape[:, :, :-1, 1:-1:2] = (mu[:, :, :-1, :-1] + mu[:, :, :-1, 1:] + mu[:, :, 1:, :-1]) / 3
    mu_reshape[:, :, :-1, -1] = (mu[:, :, :-1, -1] + mu[:, :, 1:, -1]) / 2
    mu_reshape[:, :, -1, 1:-1:2] = (mu[:, :, -1, :-1] + mu[:, :, -1, 1:]) / 2
    mu_reshape[:, :, -1, -1] = mu[:, :, -1, -1]
    mu = mu_reshape.reshape((N, 2, -1))
    return mu

def generalized_laplacian2D_torch(face, vertex, h, w, device, batch_size):
    """
    Inputs:
        face : m x 3 index of triangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
        mu : N x 2 x m Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
        abc : vectors containing the coefficients alpha, beta and gamma (m, 3)
        area : float, area of every triangles in the mesh
    """
    f0, f1, f2 = face[:, 0], face[:, 1], face[:, 2]
    N = batch_size

    vertex = vertex.to(dtype = torch.float32)

    uxv0 = (vertex[f1,1] - vertex[f2,1]).reshape((1, h-1, (w-1)*2))
    uyv0 = (vertex[f2,0] - vertex[f1,0]).reshape((1, h-1, (w-1)*2))
    uxv1 = (vertex[f2,1] - vertex[f0,1]).reshape((1, h-1, (w-1)*2))
    uyv1 = (vertex[f0,0] - vertex[f2,0]).reshape((1, h-1, (w-1)*2))
    uxv2 = (vertex[f0,1] - vertex[f1,1]).reshape((1, h-1, (w-1)*2))
    uyv2 = (vertex[f1,0] - vertex[f0,0]).reshape((1, h-1, (w-1)*2))

    xx00 = uxv0 * uxv0
    xy00 = uxv0 * uyv0
    yy00 = uyv0 * uyv0
    xx11 = uxv1 * uxv1
    xy11 = uxv1 * uyv1
    yy11 = uyv1 * uyv1
    xx22 = uxv2 * uxv2
    xy22 = uxv2 * uyv2
    yy22 = uyv2 * uyv2

    xx10 = uxv1 * uxv0
    xy10 = uxv1 * uyv0
    xy01 = uxv0 * uyv1
    yy10 = uyv1 * uyv0

    xx21 = uxv2 * uxv1
    xy21 = uxv2 * uyv1
    xy12 = uxv1 * uyv2
    yy21 = uyv2 * uyv1

    xx02 = uxv0 * uxv2
    xy02 = uxv0 * uyv2
    xy20 = uxv2 * uyv0
    yy02 = uyv0 * uyv2


    area = (1/(h-1)) * (1/(w-1)) / 2

    nv00 = torch.zeros((N, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)
    nv11 = torch.zeros((N, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)
    nv22 = torch.zeros((N, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)
    nv01 = torch.zeros((N, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)
    nv12 = torch.zeros((N, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)
    nv20 = torch.zeros((N, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)

    def generalized_laplacian2D(mu):
        mu = mu.reshape((N, 2, h-1, (w-1)*2))
        mu_sqr = torch.clamp(torch.sum(mu**2, dim=1), min=0, max=1-1e-6)
        af = (1 - 2 * mu[:, 0] + mu_sqr) / (1 - mu_sqr)
        bf = -2 * mu[:, 1] / (1 - mu_sqr)
        gf = (1 + 2 * mu[:, 0] + mu_sqr) / (1 - mu_sqr)

        vmu = torch.zeros((N, 2, h-1 + 2, (w-1)*2 + 4), device=device, dtype = torch.float32)
        vmu[:, :, 1:-1, 2:-2] = mu
        vmu = (vmu[:, :, :-1, 1:-2:2] + vmu[:, :, :-1, 2:-1:2] + vmu[:, :, :-1, 3::2] + vmu[:, :, 1:, :-3:2] + vmu[:, :, 1:, 1:-2:2] + vmu[:, :, 1:, 2:-1:2]) / 6
        vmu_sqr = torch.clamp(torch.sum(vmu**2, dim=1), min=0, max=1-1e-6).unsqueeze(1)
        vmu_weight = 1 - vmu_sqr

        nv00[:, 1:-1, 2:-2] = af * xx00 + 2 * bf * xy00 + gf * yy00
        nv11[:, 1:-1, 2:-2] = af * xx11 + 2 * bf * xy11 + gf * yy11
        nv22[:, 1:-1, 2:-2] = af * xx22 + 2 * bf * xy22 + gf * yy22
        nv01[:, 1:-1, 2:-2] = af * xx10 + bf * (xy10 + xy01) + gf * yy10
        nv12[:, 1:-1, 2:-2] = af * xx21 + bf * (xy21 + xy12) + gf * yy21
        nv20[:, 1:-1, 2:-2] = af * xx02 + bf * (xy02 + xy20) + gf * yy02

        A = torch.zeros((N, 7, h, w), device=device, dtype = torch.float32)
        A[:, 0, :, :] = nv01[:, :-1, 1:-2:2] + nv01[:, :-1, 2:-1:2]
        A[:, 1, :, :] = nv12[:, :-1, 2:-1:2] + nv12[:, :-1, 3::2]
        A[:, 2, :, :] = nv20[:, :-1, 1:-2:2] + nv20[:, 1:, :-3:2]
        A[:, 3, :, :] = nv00[:, :-1, 1:-2:2] + nv11[:, :-1, 2:-1:2] + nv22[:, :-1, 3::2] + nv22[:, 1:, :-3:2] + nv11[:, 1:, 1:-2:2] + nv00[:, 1:, 2:-1:2]
        A[:, 4, :, :] = nv20[:, :-1, 3::2] + nv20[:, 1:, 2:-1:2]
        A[:, 5, :, :] = nv12[:, 1:, :-3:2] + nv12[:, 1:, 1:-2:2]
        A[:, 6, :, :] = nv01[:, 1:, 1:-2:2] + nv01[:, 1:, 2:-1:2]
        A = -A/(2*area)

        assert mu.shape[0] == N, "Error, mu.shape[0] != batch_size!"
        return A, area, vmu_weight
    return generalized_laplacian2D

def mu2A_torch(h, w, device, batch_size):
    face, vertex = meshgen(h, w)
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)
    lapla = generalized_laplacian2D_torch(face, vertex, h, w, device, batch_size)

    def mu2A(mu_pad0):
        """
        Inputs:
            mu_pad0 : (N, 2, h, w), the elements in the right-most column and bottom row are padded by 0.
        Outputs:
            A : 2-dimensional generalized laplacian operator (N, 7, h, w)
        """
        with torch.no_grad():
            mu = batch_mu2reshape_torch(mu_pad0, device)
            A, area, vmu_weight = lapla(mu)
            #A_avg = torch.mean(torch.abs(A), dim=1).unsqueeze(1)
            #A = A/A_avg
        return A, vmu_weight
    return mu2A

def S_Jzdz(mapping, mu):
    """
    Inputs:
        mapping: (N, 2, h, w), torch tensor
        mu : m x 1 Beltrami coefficients, mu=(h-1)*(w-1)*2
    Outputs:
        S : torch float number
    """
    N, C, H, W = mapping.shape
    device = mapping.device
    face, vertex = meshgen(H, W)
    _, abc, _ = generalized_laplacian2D(face, vertex, mu, H, W)
    abc = torch.from_numpy(abc).to(device=device)
    af = abc[:, 0]
    bf = abc[:, 1]
    gf = abc[:, 2]

    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]

    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]

    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi

    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]

    sjsi = sj - si
    sksi = sk - si

    a = (sjsi * hkhi + sksi * hihj) / area / 2;
    b = (sjsi * gigk + sksi * gjgi) / area / 2;
    S = torch.mean(af*a*a+2*bf*a*b+gf*b*b)    # Using mean because the area of all the triangles are the same.

    return S

def mesh_area(mapping):
    """
    Inputs:
        mapping: (N, 2, h, w), torch tensor
    Outputs:
        S : torch float number
    """
    N, C, H, W = mapping.shape
    device = mapping.device
    face, _ = meshgen(H, W)
    face = torch.from_numpy(face).to(device=device)

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]

    ti = mapping[:, face[:, 0], 1]
    tj = mapping[:, face[:, 1], 1]
    tk = mapping[:, face[:, 2], 1]

    sjsi = sj - si
    sksi = sk - si
    tjti = tj - ti
    tkti = tk - ti

    area = (sjsi * tkti - sksi * tjti) / 2
    S = torch.sum(torch.abs(area))
    return S 

def bc_metric(mapping):
    """
    Inputs:
        mapping: (N, 2, h, w), torch tensor
    Outputs:
        mu: (N, 2, (h-1)*(w-1)*2), torch tensor
    """
    # The three input variables are pytorch tensors.
    N, C, H, W = mapping.shape
    device = mapping.device
    face, vertex = meshgen(H, W)
    face = torch.from_numpy(face).to(device=device)
    vertex = torch.from_numpy(vertex).to(device=device)

    gi = vertex[face[:, 0], 0]
    gj = vertex[face[:, 1], 0]
    gk = vertex[face[:, 2], 0]
    
    hi = vertex[face[:, 0], 1]
    hj = vertex[face[:, 1], 1]
    hk = vertex[face[:, 2], 1]
    
    gjgi = gj - gi
    gkgi = gk - gi
    hjhi = hj - hi
    hkhi = hk - hi
    
    area = (gjgi * hkhi - gkgi * hjhi) / 2

    gigk = -gkgi
    hihj = -hjhi

    mapping = mapping.view((N, C, -1))
    mapping = mapping.permute(0, 2, 1)

    si = mapping[:, face[:, 0], 0]
    sj = mapping[:, face[:, 1], 0]
    sk = mapping[:, face[:, 2], 0]
    
    ti = mapping[:, face[:, 0], 1]
    tj = mapping[:, face[:, 1], 1]
    tk = mapping[:, face[:, 2], 1]
    
    sjsi = sj - si
    sksi = sk - si
    tjti = tj - ti
    tkti = tk - ti
    
    a = (sjsi * hkhi + sksi * hihj) / area / 2;
    b = (sjsi * gigk + sksi * gjgi) / area / 2;
    c = (tjti * hkhi + tkti * hihj) / area / 2;
    d = (tjti * gigk + tkti * gjgi) / area / 2;
    
    down = (a+d)**2 + (c-b)**2 + 1e-8
    up_real = (a**2 - d**2 + c**2 - b**2)
    up_imag = 2*(a*b+c*d)
    real = up_real / down
    imag = up_imag / down

    mu = torch.stack((real, imag), dim=1)
    return mu


