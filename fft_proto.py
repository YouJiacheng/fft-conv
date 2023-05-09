
import numpy as np
from numpy.fft import fft as np_fft

x = np.zeros(4096, dtype=np.complex64)
x.real = np.random.rand(4096)
x.imag = np.random.rand(4096)

def dft_matrix(n: int):
    i, j = np.ogrid[0:n, 0:n]
    return np.exp((i * j) * (-2j * np.pi / n))

F = dft_matrix(16)
assert np.allclose(F @ x[:16], np_fft(x[:16]))

def grid(*sizes: int) -> list[np.ndarray]:
    return list(np.mgrid[tuple(slice(size) for size in sizes)])

def einsum(expr: str, /, *operands: np.ndarray) -> np.ndarray:
    return np.einsum(expr, *operands)

def pease_twiddle(l: int, m: int, n: int):
    i, j, k = grid(l, m, n)
    return np.exp((i * k) * (-2j * np.pi / (l * n))).flatten()

def pease_fft_256(x: np.ndarray):
    assert x.size == 256
    N = 256
    R = 16

    x = x.reshape((R, N // R)).T.flatten()
    x = einsum('ij,bj->bi', F.T, x.reshape((N // R, R))).flatten()
    x = x * pease_twiddle(N // R, 1, R)

    x = x.reshape((R, N // R)).T.flatten()
    x = einsum('ij,bj->bi', F.T, x.reshape((N // R, R))).flatten()
    x = x * pease_twiddle(N // R // R, R, R) # all 1's

    return x.reshape((R, R)).T.flatten()

def pease_fft_4096(x: np.ndarray):
    assert x.size == 4096
    N = 4096
    R = 16

    x = x.reshape((R, N // R)).T.flatten()
    x = einsum('ij,bj->bi', F.T, x.reshape((N // R, R))).flatten()
    x = x * pease_twiddle(N // R, 1, R)

    x = x.reshape((R, N // R)).T.flatten()
    x = einsum('ij,bj->bi', F.T, x.reshape((N // R, R))).flatten()
    x = x * pease_twiddle(N // R // R, R, R)

    x = x.reshape((R, N // R)).T.flatten()
    x = einsum('ij,bj->bi', F.T, x.reshape((N // R, R))).flatten()
    x = x * pease_twiddle(N // R // R // R, R * R, R) # all 1's

    return x.reshape((R, R, R)).T.flatten()

print(np.allclose(pease_fft_256(x[:256]), np_fft(x[:256])))
print(np.allclose(pease_fft_4096(x), np_fft(x)))


def cooley_twiddle(l: int, m: int, n: int):
    i, j, k = grid(l, m, n)
    return np.exp((j * k) * (-2j * np.pi / (m * n)))

def cooley_fft_256(x: np.ndarray):
    assert x.size == 256
    N = 256
    R = 16

    x = x.reshape((R, R))

    x = einsum('ij,jb->ib', F.T, x)
    x = x * cooley_twiddle(1, R, N // R).reshape((R, R))

    x = einsum('ij,bj->bi', F.T, x)
    x = x * cooley_twiddle(R, R, N // R // R).reshape((R, R)) # all 1's

    return x.T.flatten()

def cooley_fft_4096(x: np.ndarray):
    assert x.size == 4096
    N = 4096
    R = 16

    x = x.reshape((R, R, R))

    x = einsum('ij,jbc->ibc', F.T, x)
    x = x * cooley_twiddle(1, R, N // R).reshape((R, R, R))

    x = einsum('ij,bjc->bic', F.T, x)
    x = x * cooley_twiddle(R, R, N // R // R).reshape((R, R, R))

    x = einsum('ij,bcj->bci', F.T, x)
    x = x * cooley_twiddle(R * R, R, N // R // R // R).reshape((R, R, R)) # all 1's

    return x.T.flatten()

print(np.allclose(cooley_fft_256(x[:256]), np_fft(x[:256])))
print(np.allclose(cooley_fft_4096(x), np_fft(x)))
