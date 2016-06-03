s.boot
({ FSM("
print(np.array([1,2,3,4]))
import sys
sys.stdout.flush()
") }.play)
