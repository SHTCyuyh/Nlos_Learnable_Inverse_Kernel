import torch
from scipy import io
import scipy.sparse as ssp
import numpy as np
import matplotlib.pyplot as plt

def interpolate(grid, lin_ind_frustrum, voxel_coords, device_id):
    """ linear interpolation for frequency-wavenumber migration
        adapted from https://github.com/vsitzmann/deepvoxels/blob/49369e243001658ccc8ba3be97d87c85273c9f15/projection.py
    """

    depth, width, height = grid.shape

    lin_ind_frustrum = lin_ind_frustrum.long()

    x_indices = voxel_coords[1, :]
    y_indices = voxel_coords[2, :]
    z_indices = voxel_coords[0, :]

    mask = ((x_indices < 0) | (y_indices < 0) | (z_indices < 0) |
            (x_indices > width-1) | (y_indices > height-1) | (z_indices > depth-1)).to(device_id)

    x0 = x_indices.floor().long()
    y0 = y_indices.floor().long()
    z0 = z_indices.floor().long()

    x0 = torch.clamp(x0, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    z0 = torch.clamp(z0, 0, depth - 1)
    z1 = (z0 + 1).long()
    z1 = torch.clamp(z1, 0, depth - 1)

    x_indices = torch.clamp(x_indices, 0, width - 1)
    y_indices = torch.clamp(y_indices, 0, height - 1)
    z_indices = torch.clamp(z_indices, 0, depth - 1)

    x = x_indices - x0.float()
    y = y_indices - y0.float()
    z = z_indices - z0.float()

    output = torch.zeros(height * width * depth).to(device_id)
    tmp1 = grid[z0, x0, y0] * (1 - z) * (1 - x) * (1 - y)
    tmp2 = grid[z1, x0, y0] * z * (1 - x) * (1 - y)
    output[lin_ind_frustrum] = tmp1 + tmp2

    output = output * (1 - mask.float())
    output = output.contiguous().view(depth, width, height)

    return output


def roll_n(X, axis, n):
    """ circular shift function """

    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(x):
    real, imag = torch.unbind(x, -1)

    if real.ndim > 3:
        dim_start = 2
    else:
        dim_start = 0

    for dim in range(dim_start, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def ifftshift(x):
    real, imag = torch.unbind(x, -1)

    if real.ndim > 3:
        dim_stop = 1
    else:
        dim_stop = -1

    for dim in range(len(real.size()) - 1, dim_stop, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def compl_mul(X, Y):
    """ complex multiplication for pytorch; real and imaginary parts are
        stored in the last channel of the arrays
        see https://discuss.pytorch.org/t/aten-cuda-implementation-of-complex-multiply/17215/2
    """

    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def conj(x):
    # complex conjugation for pytorch

    tmp = x.clone()
    tmp[:, :, :, :, :, 1] = tmp[:, :, :, :, :, 1] * -1
    return tmp


def fk(meas, width, mrange):
    """ perform f--k migration """

    device = meas.device
    meas = meas.squeeze()
    width = torch.FloatTensor([width]).to(device)
    mrange = torch.FloatTensor([mrange]).to(device)

    N = meas.size()[1]//2  # spatial resolution
    M = meas.size()[0]//2  # temporal resolution
    data = torch.sqrt(torch.clamp(meas, 0))

    M_grid = torch.arange(-M, M).to(device)
    N_grid = torch.arange(-N, N).to(device)
    [z, x, y] = torch.meshgrid(M_grid, N_grid, N_grid)
    z = (z.type(torch.FloatTensor) / M).to(device)
    x = (x.type(torch.FloatTensor) / N).to(device)
    y = (y.type(torch.FloatTensor) / N).to(device)

    # pad data
    tdata = data

    # fourier transform
    if tdata.ndim > 3:
        tdata = fftshift(tdata.fft(3))
    else:
        tdata = fftshift(tdata.rfft(3, onesided=False))

    tdata_real, tdata_imag = torch.unbind(tdata, -1)

    # interpolation coordinates
    z_interp = torch.sqrt(abs((((N * mrange) / (M * width * 4))**2) *
                              (x**2 + y**2) + z**2))
    coords = torch.stack((z_interp.flatten(), x.flatten(), y.flatten()), 0)
    lin_ind = torch.arange(z.numel()).to(device)
    coords[0, :] = (coords[0, :] + 1) * M
    coords[1, :] = (coords[1, :] + 1) * N
    coords[2, :] = (coords[2, :] + 1) * N

    # run interpolation
    tvol_real = interpolate(tdata_real, lin_ind, coords, device)
    tvol_imag = interpolate(tdata_imag, lin_ind, coords, device)
    tvol = torch.stack((tvol_real, tvol_imag), -1)

    # zero out redundant spectrum
    x = x[:, :, :, None]
    y = y[:, :, :, None]
    z = z[:, :, :, None]
    tvol = tvol * abs(z) / torch.clamp(torch.sqrt(abs((((N * mrange) / (M * width * 4))**2) *
                                       (x**2 + y**2)+z**2)), 1e-8)
    tvol = tvol * (z > 0).type(torch.FloatTensor).to(device)

    # inverse fourier transform and crop
    tvol = ifftshift(tvol).ifft(3).squeeze()
    geom = tvol[:, :, :, 0]**2 + tvol[:, :, :, 1]**2
    geom = geom[None, None, :, :, :]

    return geom

def circshift(u, shiftnums):
    h, w, d = u.shape
    shiftnum1, shiftnum2, shiftnum3 = shiftnums
    # print(shiftnum1, shiftnum2, shiftnum3) 
    # print(h,w,d)

    # u = torch.stack((u[(h-shiftnum1):,:,:], u[:(h-shiftnum1),:,:]), dim=0)
    # u = torch.stack((u[:,(w-shiftnum2):,:], u[:,:(w-shiftnum2),:]), dim=1)
    # u = torch.stack((u[:,:,(d-shiftnum3):], u[:,:,:(d-shiftnum3)]), dim=2)

    u = torch.cat((u[(h-shiftnum1):,:,:], u[:(h-shiftnum1),:,:]), dim=0)
    u = torch.cat((u[:,(w-shiftnum2):,:], u[:,:(w-shiftnum2),:]), dim=1)
    u = torch.cat((u[:,:,(d-shiftnum3):], u[:,:,:(d-shiftnum3)]), dim=2)

    # print(u.shape)
    return u


# def definePsf(U,V,slope):

#     x = torch.linspace(-1,1,2*U)
#     y = torch.linspace(-1,1,2*U)
#     z = torch.linspace(0,2,2*V)
#     [grid_z,grid_y,grid_x] = torch.meshgrid(z,y,x)

#     psf = torch.abs(((4*slope)**2) * (grid_x**2 + grid_y**2) - grid_z)
#     # psf = (psf.min(dim=0).values).repeat([2*V, 1, 1])
#     # print(psf.min(dim=0).values.shape)
#     # print((psf.min(dim=0).values).repeat([2*V, 1, 1]).shape)
#     psf = (psf == (psf.min(dim=0).values).repeat([2*V, 1, 1])).double()
 
#     psf = psf / torch.sum(psf[:,U,U])
#     # print(torch.norm(psf))
#     psf = psf / torch.norm(psf)
#     psf = circshift(psf, [0, U, U])
#     return psf

def definePsf(sptial_grid, temprol_grid, slope):
    # slop is time_range / wall_size
    N = sptial_grid
    M = temprol_grid

    # -1 to 1
    x_2N = np.arange(2 * sptial_grid, dtype=np.float32)
    x_2N = x_2N / (2 * sptial_grid - 1) * 2 - 1

    # here, x and y are symetric
    # it doesn't mater y is postive or negative
    y_2N = x_2N

    # 0 to 2
    z_2M = np.arange(2 * temprol_grid, dtype=np.float32)
    z_2M = z_2M / (2 * temprol_grid - 1) * 2

    # grid axis, also in hxwxt
    # that's why x is the second axis
    # y is the first axis
    [gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(x_2N, y_2N, z_2M)

    # dst
    a_2Nx2NX2M = (4 * slope) ** 2 * (gridx_2Nx2Nx2M ** 2 + gridy_2Nx2Nx2M ** 2) - gridz_2Nx2Nx2M
    b_2Nx2NX2M = np.abs(a_2Nx2NX2M)

    # min data
    c_2Nx2NX2M = np.min(b_2Nx2NX2M, axis=2, keepdims=True)

    # should be a ellipse
    d_2Nx2NX2M = np.abs(b_2Nx2NX2M - c_2Nx2NX2M) < 1e-8
    d_2Nx2NX2M = d_2Nx2NX2M.astype(np.float32)

    # norm
    e_2Nx2NX2M = d_2Nx2NX2M / np.sqrt(np.sum(d_2Nx2NX2M))

    # shift
    f1_2Nx2NX2M = np.roll(e_2Nx2NX2M, shift=N, axis=0)
    f2_2Nx2NX2M = np.roll(f1_2Nx2NX2M, shift=N, axis=1)

    psf_2Mx2Nx2N = np.transpose(f2_2Nx2NX2M, [2, 0, 1])

    return psf_2Mx2Nx2N

def resamplingOperator(M):
    M = M
    row = M ** 2
    col = M
    assert 2 ** int(np.log2(M)) == M

    # 1 to M^2
    x = np.arange(row, dtype=np.float32)
    x = x + 1

    rowidx = np.arange(row)
    # 0 to M-1
    colidx = np.ceil(np.sqrt(x)) - 1
    data = np.ones_like(rowidx, dtype=np.float32)
    mtx1 = ssp.csr_matrix((data, (rowidx, colidx)), shape=(row, col), dtype=np.float32)
    mtx2 = ssp.spdiags(data=[1.0 / np.sqrt(x)], diags=[0], m=row, n=row)

    mtx = mtx2.dot(mtx1)
    # mtxi = np.transpose(mtx)
    K = int(np.log2(M))
    for _ in np.arange(K):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2])
        # mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    mtxi = np.transpose(mtx)

    return mtx.toarray(), mtxi.toarray()

def gaussianwin(L, alpha):
    N = L - 1
    Nhalf = N / 2.0
    n_k = np.arange(N + 1, dtype=np.float32) - Nhalf
    w_k = np.exp(-0.5 * (alpha * n_k / Nhalf) ** 2)

    return w_k

def waveconvparam(bin_resolution, virtual_wavelength, cycles):
    c = 3e8;
    s_z = bin_resolution * c
    samples = int(round(cycles * virtual_wavelength / (bin_resolution * c)))
    num_cycles = samples * s_z / virtual_wavelength
    sigma = 0.3

    # generate sin/cos signals
    grids_k = np.arange(samples, dtype=np.float32) + 1
    sin_wave_k = np.sin(2 * np.pi * (num_cycles * grids_k) / samples)
    cos_wave_k = np.cos(2 * np.pi * (num_cycles * grids_k) / samples)

    # window = single(gausswin(samples, 1/sigma));
    window = gaussianwin(samples, 1.0 / sigma)
    virtual_sin_wave_k = sin_wave_k * window
    virtual_cos_wave_k = cos_wave_k * window

    return virtual_cos_wave_k, virtual_sin_wave_k

def waveconv(bin_resolution, virtual_wavelength, cycles, data_txhxw):
    c = 3e8
    s_z = bin_resolution * c
    samples = int(round(cycles * virtual_wavelength / (bin_resolution* c)))
    num_cycles = samples *  s_z/ virtual_wavelength
    sigma = 0.3

    # generate sin/cos signals
    grids_k = np.arange(samples, dtype=np.float32) + 1
    sin_wave_k = np.sin(2 * np.pi * (num_cycles * grids_k) / samples)
    cos_wave_k = np.cos(2 * np.pi * (num_cycles * grids_k) / samples)

    # window = single(gausswin(samples, 1/sigma));
    window = gaussianwin(samples, 1.0 / sigma)
    virtual_sin_wave_k = sin_wave_k * window
    virtual_cos_wave_k = cos_wave_k * window

    wave_sin = np.zeros(data_txhxw.shape)
    wave_cos = np.zeros(data_txhxw.shape)

    # conv
    M, N, _ = data_txhxw.shape
    for i in range(N):
        for j in range(N):
            data_t = data_txhxw[:, i, j].cpu().numpy()
            real = np.convolve(data_t, v=virtual_sin_wave_k, mode='same')
            image = np.convolve(data_t, v=virtual_cos_wave_k, mode='same')
            wave_sin[:, i, j] = real
            wave_cos[:, i, j] = image

    return wave_cos, wave_sin

def pf(meas, wall_size, bin_resolution):

    print(meas.shape, wall_size, bin_resolution)
    c = 3e8
    width = wall_size / 2
    M, _, N = meas.shape

    range = M * c * bin_resolution

    psf = definePsf(N, M, width/range)

    bp_psf = np.conjugate(np.fft.fftn(psf))

    mtx, mtxi = resamplingOperator(M)
    mtx = torch.from_numpy(mtx.astype(np.float32))
    mtxi = torch.from_numpy(mtxi.astype(np.float32))

    s_lamda_limit = wall_size / (N - 1)
    sampling_coeff = 2
    virtual_wavelength = sampling_coeff * (s_lamda_limit * 2)
    cycles = 5

    phasor_data_cos, phasor_data_sin = waveconv(bin_resolution, virtual_wavelength, cycles, meas)
    phasor_data_cos = torch.Tensor(phasor_data_cos)
    phasor_data_sin = torch.Tensor(phasor_data_sin)

    phasor_tdata_cos = torch.zeros(2*M, 2*N, 2*N)
    phasor_tdata_sin = torch.zeros(2*M, 2*N, 2*N)
    phasor_tdata_cos[0:M,0:N,0:N] = (mtx.mm(phasor_data_cos.reshape([M, N*N]))).reshape([M, N, N])
    phasor_tdata_sin[0:M,0:N,0:N] = (mtx.mm(phasor_data_sin.reshape([M, N*N]))).reshape([M, N, N])

    phasor_tdata_sin = phasor_tdata_sin.numpy()
    phasor_tdata_cos = phasor_tdata_cos.numpy()
    tvol_phasorbp_sin = np.fft.ifftn(np.fft.fftn(phasor_tdata_sin) * bp_psf)
    tvol_phasorbp_sin = tvol_phasorbp_sin[0:M,0:N,0:N]
    phasor_tdata_cos = np.fft.ifftn(np.fft.fftn(phasor_tdata_cos) * bp_psf) 
    phasor_tdata_cos = phasor_tdata_cos[0:M,0:N,0:N]

    # phasor_tdata_cos = torch.from_numpy(phasor_tdata_cos)
    # tvol_phasorbp_sin = torch.from_numpy(tvol_phasorbp_sin)

    tvol = np.sqrt(np.power(tvol_phasorbp_sin,2) + np.power(phasor_tdata_cos, 2))
    mtxi = mtxi.numpy()
    vol  = (np.matmul(mtxi,tvol.astype(np.float32).reshape([M, N*N]))).reshape([M, N, N])
    # vol  = vol.real
    vol[vol < 0] = 0    
    print("PF done.")
    return torch.Tensor(vol).cuda() 


if __name__ == '__main__':
    import cv2
    from einops import rearrange
    test_phasor = True
    path = '/data2/nlospose/pose_v2_noise/pose_00/train/meas/person00-00009.hdr'
    if test_phasor:
        data = cv2.imread(path, -1)
        data = data / np.max(data)
        meas = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        meas = meas / np.max(meas)
        meas = rearrange(meas, '(t h) w ->t h w', t=600)
        meas = meas[:512, :, :]
        K = 1
        for i in range(K):
             meas = meas[::2, :, :] + meas[1::2, :, :]
             meas = meas[:, ::2, :] + meas[:, 1::2, :]
             meas = meas[:, :, ::2] + meas[:, :, 1::2]
    # print(meas.shape)
    meas = torch.from_numpy(meas)
    x_pf = pf(meas, 2, 0.02/3e8)
    # x_pf_npy = x_pf.cpu().data.numpy().squeeze()
    x_pf_npy = x_pf.cpu().numpy()
    # trim any amplified noise at the very end of the volume
    # x_npy[-15:, :, :] = 0

    plt.imshow(np.max(x_pf_npy, axis=0), cmap='gray')
    plt.savefig('./test.png')