import numpy as np

def out(dst, src):
    if type(dst) is list:
        for i in range(0, len(dst)):
            dst[i][:] = src[i]
    elif type(dst) is array:
        dst[:] = src[:]
    else:
        raise "invalid output type"

def to_polar(x):
    return np.abs(x) + 1j * np.angle(x)

def from_polar(x):
    return x.real * np.exp(1j * x.imag)
