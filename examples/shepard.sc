// transform any sound into a falling shephard tone
// you'll need some sound going into SuperCollider's first 2 inputs.
(s.waitForBoot {
    var buf = { Buffer.alloc(s, 512) }.dup;
    var hop = 1/4;

    PyOnce("
        pv = PhaseVocoder(hop)

        def fn(x, time):
            x = pv.forward(x)
            idx = indices(x.shape)[1]
            x = pv.shift(x, lambda y:
                y * (0.8 + mod(-time + 0.1*idx, 10)*0.045))
            x = pv.backward(x)
            return x
    ", (hop:hop));

    s.freeAll;
    {
        var in = AudioIn.ar([1, 2]);
        var x = FFT(buf.collect(_.bufnum), in, hop);
        Py("
            out(x, fn(array(x), time))
        ", (x:x, time:Sweep.kr));
        Out.ar(0, IFFT(x));
    }.play(s);
})
