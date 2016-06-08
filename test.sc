s.boot
b = Buffer.alloc(s, 100);
({ FSM("
b[0] += np.arange(100)
print(b)
", (b:b)) }.play)
b.plot
