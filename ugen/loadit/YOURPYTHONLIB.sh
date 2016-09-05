#!/bin/bash
# Hi! let's try to find out where your python lib actually is!
echo `python3-config --configdir`/lib`python3-config --libs | sed -e 's/ /\n/'g | grep python | sed -e 's/^-l//'`.so
