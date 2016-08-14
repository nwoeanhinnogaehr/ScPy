#include "sc_objs.h"
#include <iostream>

void
sanitize(int len, float* data)
{
    for (int i = 0; i < len; i++) {
        if (isnan(data[i]) || isinf(data[i]))
            data[i] = 0;
    }
}

FloatBufferObject::FloatBufferObject(int samples, int channels, int frames,
                                     float* data)
  : BufferObject(samples, channels, frames, data, NPY_FLOAT)
{
}

FloatBufferObject::~FloatBufferObject()
{
}

void
FloatBufferObject::send()
{
    memcpy(PyArray_DATA((PyArrayObject*)_obj), _data,
           _channels * _frames * sizeof(float));
}

void
FloatBufferObject::recv()
{
    memcpy(_data, PyArray_DATA((PyArrayObject*)_obj),
           _channels * _frames * sizeof(float));
    sanitize(_samples, (float*)PyArray_DATA((PyArrayObject*)_obj));
}

ComplexBufferObject::ComplexBufferObject(int samples, int channels, int frames,
                                         complex<float>* data)
  : BufferObject(samples, channels, frames, data, NPY_CFLOAT)
{
}

ComplexBufferObject::~ComplexBufferObject()
{
}

void
ComplexBufferObject::send()
{
    // supercollider stores the dc and nyquist as floats at the
    // beginning of the array
    // to simplify working with the fft output, make them complex and
    // move the nyquist
    // to the end.
    complex<float>* pyData =
      reinterpret_cast<complex<float>*>(PyArray_DATA((PyArrayObject*)_obj));
    memcpy(pyData + 1, _data + 1, (_samples - 2) * sizeof(complex<float>));
    pyData[0] = _data[0].real();
    pyData[_samples - 1] = _data[0].imag();
}

void
ComplexBufferObject::recv()
{
    complex<float>* pyData =
      reinterpret_cast<complex<float>*>(PyArray_DATA((PyArrayObject*)_obj));
    memcpy(_data + 1, pyData + 1, (_samples - 2) * sizeof(complex<float>));
    _data[0] = complex<float>(pyData[0].real(), pyData[_samples - 1].real());
    sanitize(_samples * 2 - 2, (float*)PyArray_DATA((PyArrayObject*)_obj));
}

ArrayObject::ArrayObject(vector<Object*> objs)
{
    _objs = objs;
    _obj = PyList_New(objs.size());
    for (size_t i = 0; i < objs.size(); i++) {
        PyList_SetItem(_obj, i, objs[i]->pyObject());
    }
}

void
ArrayObject::send()
{
    // TODO if the array items have been reassigned in python, this can
    // segfault!
    for (Object* obj : _objs) {
        obj->send();
    }
}

void
ArrayObject::recv()
{
    for (Object* obj : _objs) {
        obj->recv();
    }
}

ConstObject::ConstObject(float value)
{
    _obj = Py_BuildValue("f", value);
}

ControlUGenObject::ControlUGenObject(float* ptr)
{
    _ptr = ptr;
    long dims[1] = { 1 };
    _obj = PyArray_SimpleNew(1, dims, NPY_FLOAT);
}

void
ControlUGenObject::send()
{
    *(float*)PyArray_DATA((PyArrayObject*)_obj) = *_ptr;
}
