s.boot;
(
    b = { Buffer.alloc(s,128,1) }.dup;
    FSMInit("
def fn(x):
    return np.conj(x)
    ");
)
(
    { var in, chain;
        in = AudioIn.ar([1,2]);
        chain = FFT(b.collect(_.bufnum), in);
        FSM("
l[:] = fn(l)
r[:] = fn(r)
        ", (l:chain[0], r:chain[1]));
        Out.ar(0, IFFT(chain));
    }.play(s)
)
