from numpy import *

def out(dst, src):
    if type(dst) is list:
        for i in range(0, len(dst)):
            dst[i][:] = src[i]
    elif type(dst) is array:
        dst[:] = src[:]
    else:
        raise "invalid output type"
