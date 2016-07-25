s.boot; // run this line first

// then, execute the block to perform setup.
(
    b = { Buffer.alloc(s,32768,1) }.dup;

    // this runs the enclosed python code once,
    // to define the processing function.
    // if you change this function, just re-evaluate
    // the block and the changes will be reflected
    // immediately.
    PyOnce("
        def fn(x, time, mouse):
            x = np.abs(x) + 1j * np.angle(x) # to polar
            x.real = np.cos(x.real*mouse)*x.real # some weird operation
            x = x.real * np.exp(1j * x.imag) # from polar
            return x
    ");
)

// finally, execute this block to start the synth.
(
    {
        var in = Saw.ar(8); // input is a 8Hz sawtooth wave
        var chain = FFT(b.collect(_.bufnum), in);
        var mouse = MouseX.kr(1, 0);
        Py("
            # call the function defined above
            y = fn(np.array(x), time, mouse)

            # write back the transformed version
            x[0][:] = y[0]
            x[1][:] = y[1]
        ", (x:chain, time: Sweep.kr, mouse:mouse));
        Out.ar(0, IFFT(chain).clip2(1));
    }.play(s)
)
