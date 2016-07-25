# ScPy
This project lets you embed Python code inside of SuperCollider programs, enabling the use of more computationally intensive vector operations in realtime than would be possible with SuperCollider alone.

## build
If your SuperCollider extensions dir is not `~/.local/share/SuperCollider/Extensions`, you will have to set `SC_EXT_DIR` before building.
```
$ cd ugen
$ make install
```

## run
Check out [test.sc](test.sc).
