s.boot;
(s.waitForBoot {
(
    var buf = { Buffer.alloc(s, 256) }.dup;
    var hop = 1/4;

//s.freeAll; // stop previous versions

    PyOnce("
        pv = PhaseVocoder(hop)

        def fn(x):
            x = pv.forward(x)
            x = pv.to_bin_offset(x)
            x.imag = -x.imag
            x = pv.from_bin_offset(x)
            x = pv.backward(x)
            return x
    ", (hop:hop));
//s.sync;
    {
        var in = AudioIn.ar([1, 2]);
        var x = FFT(buf.collect(_.bufnum), in, hop);
        Py("
            out(x, fn(array(x)))
        ", (x:x, time:Sweep.kr));
        Out.ar(0, IFFT(x));
    }.play(s);
)
})
