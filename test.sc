s.boot
b = Buffer.alloc(s, 10, 8);
({ FSM("
print(b)
", (b:b)) }.play)
