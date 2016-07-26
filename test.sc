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
            x = to_polar(x)
            x.real = cos(x.real*mouse)*x.real # some weird operation
            x = from_polar(x)
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
            out(x, fn(x, time, mouse))
        ", (x:chain, time:Sweep.kr, mouse:mouse));
        Out.ar(0, IFFT(chain).clip2(1));
    }.play(s)
)
