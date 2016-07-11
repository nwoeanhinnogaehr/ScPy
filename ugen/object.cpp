#include "object.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL FSM_ARRAY_API
#include <iostream>
#include <numpy/arrayobject.h>

using namespace std;

Object::Object(float value)
{
    _value = new float(value);
    _type = Type::Float;
    _obj = Py_BuildValue("f", value);
}

Object::Object(FloatBuffer& value)
{
    _value = new FloatBuffer(value);
    _type = Type::FloatBuffer;
    long dims[2] = { (long)value.channels, (long)value.frames };
    _obj = PyArray_SimpleNew(2, dims, NPY_FLOAT);
}

Object::Object(ComplexBuffer& value)
{
    _value = new ComplexBuffer(value);
    _type = Type::ComplexBuffer;
    long dims[2] = { (long)value.channels, (long)value.frames };
    _obj = PyArray_SimpleNew(2, dims, NPY_CFLOAT);
}

Object::Object(PyObject* obj)
{
    _obj = obj;
    throw "unimplemented";
    // not sure if this is necessary
    // TODO get type, value
}

void
Object::destroy()
{
    switch (type()) {
        case Type::Float:
            delete (float*)_value;
            break;
        case Type::FloatBuffer:
            delete (FloatBuffer*)_value;
            break;
        case Type::ComplexBuffer:
            delete (ComplexBuffer*)_value;
            break;
        case Type::Unsupported:
            break;
    }
    Py_DecRef(_obj);
}

Type
Object::type()
{
    return _type;
}

PyObject*
Object::getPyObject()
{
    return _obj;
}

float&
Object::getFloat()
{
    return *(float*)_value;
}

FloatBuffer&
Object::getFloatBuffer()
{
    return *(FloatBuffer*)_value;
}

ComplexBuffer&
Object::getComplexBuffer()
{
    return *(ComplexBuffer*)_value;
}

void
Object::send()
{
    switch (type()) {
        case Type::Float:
            break;
        case Type::FloatBuffer: {
            FloatBuffer& buf = getFloatBuffer();
            memcpy(PyArray_DATA((PyArrayObject*)_obj), buf.data,
                   buf.channels * buf.frames * sizeof(float));
            break;
        }
        case Type::ComplexBuffer: {
            ComplexBuffer& buf = getComplexBuffer();

            // supercollider stores the dc and nyquist as floats at the beginning of the array
            // to simplify working with the fft output, make them complex and move the nyquist
            // to the end.
            complex<float>* pyData = reinterpret_cast<complex<float>*>(
                    PyArray_DATA((PyArrayObject*)_obj));
            memcpy(pyData + 1, buf.data + 1,
                   (buf.samples - 2) * sizeof(complex<float>));
            pyData[0] = buf.data[0].real();
            pyData[buf.samples - 1] = buf.data[0].imag();
            break;
        }
        case Type::Unsupported:
            break;
    }
}

void
Object::recv()
{
    switch (type()) {
        case Type::Float:
            break;
        case Type::FloatBuffer: {
            FloatBuffer& buf = getFloatBuffer();
            memcpy(buf.data, PyArray_DATA((PyArrayObject*)_obj),
                   buf.channels * buf.frames * sizeof(float));
            break;
        }
        case Type::ComplexBuffer: {
            ComplexBuffer& buf = getComplexBuffer();

            // see commend above in send()
            complex<float>* pyData = reinterpret_cast<complex<float>*>(
                    PyArray_DATA((PyArrayObject*)_obj));
            memcpy(buf.data + 1, pyData + 1,
                   (buf.samples - 2) * sizeof(complex<float>));
            buf.data[0] = complex<float>(pyData[0].real(),
                    pyData[buf.samples - 1].real());
            break;
        }
        case Type::Unsupported:
            break;
    }
}
