#include "object.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL FSM_ARRAY_API
#include <iostream>
#include <numpy/arrayobject.h>

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
    _obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, getFloatBuffer().data);
}

Object::Object(ComplexBuffer& value)
{
    _value = new ComplexBuffer(value);
    _type = Type::ComplexBuffer;
    long dims[2] = { (long)value.channels, (long)value.frames };
    _obj = PyArray_SimpleNewFromData(2, dims, NPY_CFLOAT, getComplexBuffer().data);
}

Object::Object(PyObject* obj)
{
    _obj = obj;
    // TODO get type, value
}

Object::~Object()
{
    // TODO do we even need to store the value at all??
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
