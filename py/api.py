import numpy as np

sample_rate = 0 # this is set externally later

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

class PhaseVocoder:
    """
    The phase vocoder is a transformation that allows for more fine grained
    frequency analysis and manipulation than can be achieved from a basic
    fourier transform. It requires that some state is maintained between
    calls, so a PhaseVocoder object should be preserved globally.
    """
    def __init__(self, hop):
        """
        Create a new phase vocoder with a given hop size. This size should be
        the same as the STFT hop size. In SuperCollider, it is an optional
        argument to FFT() which defaults to 1/2. For optimal quality, try 1/4
        or 1/8.
        """
        self.last_phase = None
        self.sum_phase = None
        self.shape = None
        self.hop = hop

    def forward(self, x):
        """
        Performs a forwards phase vocoder transform on x. x is expected to be
        in cartesian form. An array of complex numbers is returned where the
        real part represents amplitude and the imaginary part represents
        frequency.
        """
        if self.shape != x.shape:
            self.shape = x.shape
            self.last_phase = np.zeros(x.shape)
            self.sum_phase = np.zeros(x.shape)
            self.frame_size = x.shape[-1]
            self.freq_per_bin = sample_rate / self.frame_size / 2.0
            self.step_size = self.frame_size * self.hop
            self.expect = 2.0 * np.pi * self.hop
            self.bins = np.arange(self.frame_size)
        polar = to_polar(x)
        p = polar.imag - self.last_phase
        self.last_phase = polar.imag
        p -= self.bins * self.expect
        qpd = (p / np.pi).astype(int)
        qpd = qpd + (qpd >= 0) * (qpd & 1) - (qpd < 0) * (qpd & 1)
        p -= np.pi * qpd.astype(float)
        p = p / self.hop / (2.0 * np.pi)
        p = self.bins * self.freq_per_bin + p * self.freq_per_bin
        return polar.real + p*1j

    def backward(self, x):
        """
        Performs a backwards phase vocoder transform on x. x is expected to be
        in the form returned by forward(). An array of complex numbers in
        cartesian form is returned.
        """
        if self.shape != x.shape:
            raise "wrong shape!"
        p = x.imag - self.bins * self.freq_per_bin
        p /= self.freq_per_bin
        p *= self.expect
        p += self.bins * self.expect
        self.sum_phase += p
        return from_polar(x.real + 1j*self.sum_phase)

    def shift(self, x, fn):
        """
        A shift operation, which can be used to implement various pitch shift
        operations, among many other more exotic effects.

        x is transformed by applying fn to the frequencies in x, then fn is
        applied to the expected center frequencies of the bins in x, to handle
        bin overflow.
        """
        row, col = np.indices(x.shape)

        # transform coordinates
        col = col.astype(float)
        col = fn(col*self.freq_per_bin)/self.freq_per_bin
        col = np.round(col).astype(int)
        col = np.clip(col, 0, x.shape[-1]-1)

        # transform frequencies
        x.imag = fn(x.imag)

        # remap
        y = np.zeros(x.shape, np.complex128)
        y[row, col] += x
        return y

    def to_bin_offset(self, x):
        """
        Converts frequency into bin relative form - the center of each bin
        becomes that bin's zero frequency.
        """
        return x.real + 1j*(x.imag - self.freq_per_bin*np.indices(x.shape)[1] + self.freq_per_bin/2)

    def from_bin_offset(self, x):
        """
        The inverse of to_bin_offset().
        """
        return x.real + 1j*(x.imag + self.freq_per_bin*np.indices(x.shape)[1] - self.freq_per_bin/2)
