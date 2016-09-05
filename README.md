# ScPy
This project lets you embed Python code inside of SuperCollider programs, enabling the use of more computationally intensive vector operations in realtime than would be possible with SuperCollider alone.

For all the details, read [the paper](doc/paper.pdf).

## build
You need Python 3, NumPy, make, and g++ or clang++.

If your SuperCollider extensions dir is not `~/.local/share/SuperCollider/Extensions`, you will have to set `SC_EXT_DIR` appropriately before building.
```
$ cd ugen
$ make install
```

If you are using Ubuntu you might have to do the following:
```
$ cd ugen
$ make ubuntuinstall
```

As we have had linking issues with python on Ubuntu 16.04.

## run
Check out the [examples](examples).

This was tested with SuperCollider 3.6.6 and Python 3.5.2.

For unknown reasons performance is much better with 3.6.6 than the latest SuperCollider.

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

## Contributors

    ScPy python embedded in supercollider
    Copyright (C) 2016 Noah Weninger, Abram Hindle

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
