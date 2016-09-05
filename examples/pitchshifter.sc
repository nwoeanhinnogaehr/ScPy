// Pitch Shifter with the mouse!
// you'll need some sound going into SuperCollider's first 2 inputs.
(s.waitForBoot {

    var buf = { Buffer.alloc(s, 512) }.dup;
    var hop = 1/4;

    PyOnce("
        pv = PhaseVocoder(hop)

        def fn(x):
            x = pv.forward(x)
            idx = indices(x.shape)[1]
            x = pv.shift(x, lambda y:
                y*offset)
            x = pv.backward(x)
            return x
    ", (hop:hop));

    s.freeAll;
    {
        var in = AudioIn.ar([1, 2]);
        var x = FFT(buf.collect(_.bufnum), in, hop);
        Py("
            out(x, fn(array(x)))
        ", (x:x, offset:MouseY.kr(0.01,100.0,1.0)));
        Out.ar(0, IFFT(x));
    }.play(s);

})
