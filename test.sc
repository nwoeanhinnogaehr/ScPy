s.boot
b = Buffer.alloc(s,2048,1);
({ var in, chain;
    in = LFSaw.ar(SinOsc.kr(0.5,0,10,50));
    chain = FFT(b.bufnum, in);
    FSM("b[:] = np.abs(b)", (b:chain));
    IFFT(chain);
}.play(s);)
