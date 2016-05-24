import numpy as np
import cmath

Cartesian = 0
Polar = 1
Freq = 2

class Spectrum(np.ndarray):
    def __new__(cls, input_array, format=None):
        obj = np.asarray(input_array).view(cls)
        obj.format = format
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.format = getattr(obj, 'format', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

# spectrum metadata
def binSize(s):
    return sampleRate(s) / numBins(s)
def numBins(s):
    return len(s)
def sampleRate(s):
    # TODO get this info from SC
    return 44100
def binCenters(s):
    return np.arange(numBins(s)) * binSize(s) + binSize(s) / 2

# transformations
class Transform:
    def forward(self, x):
        return x
    def backward(self, x):
        return x

    # function composition
    def __mod__(self, other):
        return DynamicTransform(
                forward = lambda x: self.forward(other.forward(x)),
                backward = lambda x: other.backward(self.backward(x)))

    def __call__(self, x, fn):
        return self.backward(fn(self.forward(x)))

class DynamicTransform(Transform):
    def __init__(self, **entries): self.__dict__.update(entries)

class Mask(Transform):
    def __init__(self, mask):
        self.mask = mask
    def forward(self, x):
        self.orig = x
        return self.mask.real*x.real + 1j*self.mask.imag*x.imag
    def backward(self, x):
        inv = (1 + 1j) - self.mask
        return self.orig.real*inv.real + 1j*self.orig.imag*inv.imag \
                + x.real*self.mask.real + 1j*x.imag*self.mask.imag

class Real(Transform):
    def __init__(self, mask=1):
        self.mask = mask
    def forward(self, x):
        self.orig = x
        return self.mask.real*x.real
    def backward(self, x):
        inv = 1 - self.mask
        return self.orig.real*inv + x*self.mask + 1j*self.orig.imag

class Imag(Transform):
    def __init__(self, mask=1):
        self.mask = mask
    def forward(self, x):
        self.orig = x
        return self.mask*x.imag
    def backward(self, x):
        inv = 1 - self.mask
        return self.orig.real + 1j*self.orig.imag*inv + 1j*x*self.mask

# note that this doesn't make any sense on non phase vocoder data. how can we enforce this?
class FreqOffset(Transform):
    def forward(self, x):
        return x.real + 1j*(x.imag - binCenters(x))
    def backward(self, x):
        return x.real + 1j*(x.imag + binCenters(x))

# these functions convert the spectrum data between various formats
def cartesian(s):
    if s.format == Cartesian:
        return s
    if s.format == Polar:
        n = s.real * np.exp(1j * s.imag)
        n.format = Polar
        return n
    if s.format == Freq:
        pass
def polar(s):
    if s.format == Cartesian:
        n = np.abs(s) + 1j * np.angle(s)
        n.format = Polar
        return n
    if s.format == Polar:
        return s
    if s.format == Freq:
        pass
def freq(s):
    if s.format == Cartesian:
        pass
    if s.format == Polar:
        pass
    if s.format == Freq:
        return s

# higher order functions for spectrum manipulation
def freqShift(s, fn, mixfn=lambda a, b: b, init=0):
    new = init*len(s)
    for i in range(0, len(s)):
        newFreq = fn(s[i].imag)
        newBin = round(newFreq / sampleRate(s))
        new[newBin] = mixfn(new[newBin], s[i].real + 1j * newFreq)
    return new
def map(s, fn):
    return fn(s) # TODO this is probably ok most of the time actually
def mapAccum(s, init, fn):
    pass
def filter(s, fn, clear=0):
    # sets bins to clear if fn is false, else leaves them alone
    pass
def filterAccum(s, init, fn, clear=0):
    pass

# move bins to new locations
# I guess we don't really need this
def remap(s, idxMap):
    return s[idxMap]

# linear interpolation, super useful
# there will be more math functions like this as well
def lerp(a, b, x):
    return a*x + b*(1-x)
def smoothstep(a, b, x):
    pass
