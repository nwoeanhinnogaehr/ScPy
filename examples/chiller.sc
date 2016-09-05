// frequency shift downwards relative to quantized input frequency
// I'm surprised the phase coherence is preserved so well with such a large buffer size.
// play music though it and be amazed!
// you'll need some sound going into SuperCollider's first 2 inputs.
(s.waitForBoot {

    var buf = { Buffer.alloc(s, 32768) }.dup;
    var hop = 1/4;

    PyOnce("
        pv = PhaseVocoder(hop)

        def fn(x):
            x = pv.forward(x)
            idx = indices(x.shape)[1]
            x = pv.shift(x, lambda y: y - idx*0.5)
            x = pv.backward(x)
            return x/2
    ", (hop:hop));

    s.freeAll;
    {
        var in = AudioIn.ar([1, 2]);
        var x = FFT(buf.collect(_.bufnum), in, hop);
        Py("
            out(x, fn(array(x)))
        ", (x:x, time:Sweep.kr));
        Out.ar(0, IFFT(x));
    }.play(s);

})
