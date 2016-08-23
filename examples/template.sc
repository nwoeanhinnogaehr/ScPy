(s.waitForBoot {
    var buf = { Buffer.alloc(s, 512) }.dup;
    var hop = 1/4;

    PyOnce("
        def fn(x):
            return x
    ", (hop:hop));

    s.freeAll;
    {
        var in = AudioIn.ar([1, 2]);
        var x = FFT(buf.collect(_.bufnum), in, hop);
        Py("
            out(x, fn(array(x)))
        ", (x:x));
        Out.ar(0, IFFT(x));
    }.play(s);
})
