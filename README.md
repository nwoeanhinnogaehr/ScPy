# ScPy
This project lets you embed Python code inside of SuperCollider programs, enabling the use of more computationally intensive vector operations in realtime than would be possible with SuperCollider alone.

For all the details, read [the paper](doc/paper.pdf).

## build
You need Python 3, make, and g++ or clang++.

If your SuperCollider extensions dir is not `~/.local/share/SuperCollider/Extensions`, you will have to set `SC_EXT_DIR` appropriately before building.
```
$ cd ugen
$ make install
```

## run
Check out the [examples](examples).

This was tested with SuperCollider 3.6.6 and Python 3.5.2.

It should work on the latest SC, but that has not been tested.

If you run into problems, please file an issue!

## docs

The interface:

```
Py(code, args=(), doneAction=0)
PyOnce(code, args=(), doneAction=2)
PyFile(filename, args=(), doneAction=0)
PyOnceFile(filename, args=(), doneAction=2)
```
Where:

* code is a `String` containing Python code, or filename is a `String` containing the name of a Python file.
* args is an `Event` mapping from Python variable names to SuperCollider values/variables.
* doneAction is a `DoneAction` which is performed after the code runs once.

The `Once` versions are different in that they are not UGens, but rather run the code immediately. They have the doneAction argument set such that they terminate after running the Python code once.

All NumPy functions are imported for you automatically. Variables defined at the top level in a Python block are accessible globally in other Python blocks. Some additional useful things which are accessible are defined in [py/api.py](py/api.py).

Currently supported SuperCollider types and corresponding Python types:

* `Float` -> `float`
* `UGen` -> `ndarray of float`
* `Buffer` -> `ndarray of float`
* `FFT` -> `ndarray of complex`
* `Array` -> `list`
