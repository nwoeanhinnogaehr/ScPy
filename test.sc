s.boot; // run this line first

// then, execute the block to perform setup.
(
    b = { Buffer.alloc(s,4096,1) }.dup;

    // this runs the enclosed python code once,
    // to define the processing function.
    // if you change this function, just re-evaluate
    // the block and the changes will be reflected
    // immediately.
    PyOnce("
        global bb # this is necessary to make it visible between calls
        bb = BackBuffer(64)
        def fn(x, time, mouse):
            x = to_polar(x)
            x.real = cos(x.real*mouse)*x.real # some weird operation
            x = from_polar(x)
            bb.push(x)
            x = bb.get(random.randint(0, 64))
            return x
    ");
)

// finally, execute this block to start the synth.
(
    {
        var in = Saw.ar(32); // input is a sawtooth wave
        var chain = FFT(b.collect(_.bufnum), in, hop:0.125);
        var mouse = [MouseX.kr(1, 0), MouseY.kr(1, 0)];
        Py("
            out(x, fn(x, time, mouse))
        ", (x:chain, time:Sweep.kr, mouse:mouse));
        Out.ar(0, IFFT(chain).clip2(1));
    }.play(s)
)
