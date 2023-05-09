from pathlib import Path
import cupy as cp

directory = Path(__file__).parent
kers = ('digit_reversed_cooley_tukey_16_16_16',)
with open(directory / 'fft.cu') as f:
    code = f.read()
fft_module = cp.RawModule(code=code, options=('--std=c++17', f'--include-path={directory}'), backend='nvrtc', name_expressions=kers)
fft_module.compile()
fft_kernel = fft_module.get_function('digit_reversed_cooley_tukey_16_16_16')

import numpy as np
from numpy.fft import fft, ifft

np.random.seed(42)
N = 4096
x = np.zeros(N, dtype=np.complex64)
x.real = np.random.rand(N).astype(np.float16)
x.imag = np.random.rand(N).astype(np.float16)

x = ifft(x)

x_re = cp.asarray(x.real, dtype=cp.half)
x_im = cp.asarray(x.imag, dtype=cp.half)

def dft_matrix(n: int):
    i, j = np.ogrid[0:n, 0:n]
    return np.exp((i * j) * (-2j * np.pi / n))
F = dft_matrix(16)


F_re = cp.asarray(F.real, dtype=cp.half)
F_im = cp.asarray(F.imag, dtype=cp.half)

idx = np.mgrid[0:16, 0:16]
idx_row = cp.asarray(idx[0], dtype=cp.half)
idx_col = cp.asarray(idx[1], dtype=cp.half)

fft_kernel((1,), (16 * 32,), (x_re, x_im, F_re, F_im, idx_row, idx_col))

y = np.zeros(N, dtype=np.complex64)
y.real = cp.asnumpy(x_re)
y.imag = cp.asnumpy(x_im)
ref = fft(x).reshape(16, 16, 16).T.flatten()
# ref = (dft_matrix(4096) @ x).reshape(16, 16, 16).T.flatten()
print((np.abs(ref - y).mean() / np.abs(ref).mean()) * 100)
