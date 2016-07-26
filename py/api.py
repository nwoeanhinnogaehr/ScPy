import numpy as np

def out(dst, src):
    """
    Write the data in src to the buffer dst.
    """
    if type(dst) is list:
        for i in range(0, len(dst)):
            dst[i][:] = src[i]
    elif type(dst) is array:
        dst[:] = src[:]
    else:
        raise "invalid output type"

def to_polar(x):
    """
    Convert from cartesian to polar coordinates, where the magnitude and
    phase are stored as the real and imaginary parts of a complex number,
    respectively.
    """
    return np.abs(x) + 1j * np.angle(x)

def from_polar(x):
    """
    Convert from polar to cartesian coordinates. The polar input is expected
    to have its magnitude and phase stored as the real and imaginary parts
    of a complex number, respectively.
    """
    return x.real * np.exp(1j * x.imag)

class BackBuffer:
    """
    A simple class that stores previous frames in a buffer.
    """
    def __init__(self, size):
        self.size = size
        self.items = []

    def push(self, item):
        """
        Add an item to the end of the buffer.
        If the size is exceeded, the last element will be removed.
        """
        if len(self.items) == self.size:
            del self.items[0]
        self.items.append(item)

    def get(self, idx=-1):
        """
        Retrieve an item from the buffer at a given index.
        Positive indices index forward in time, where as negative indices index
        backward in time.
        Indices outsize the bounds of the buffer will wrap around.
        """
        return self.items[idx % len(self.items)]
