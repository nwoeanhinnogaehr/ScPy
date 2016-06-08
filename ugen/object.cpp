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

Object::Object(FloatArray& value)
{
    _value = new FloatArray(value);
    _type = Type::FloatArray;
    long dims[2] = { (long)value.channels, (long)value.frames };
    _obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, getFloatArray().data);
}

Object::Object(ComplexArray& value)
{
    _value = new ComplexArray(value);
    _type = Type::FloatArray;
    // TODO obj
}

Object::Object(PyObject* obj)
{
    _obj = obj;
    // TODO get type, value
}

Object::~Object()
{
    // TODO don't leak memory
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

FloatArray&
Object::getFloatArray()
{
    return *(FloatArray*)_value;
}

ComplexArray&
Object::getComplexArray()
{
    return *(ComplexArray*)_value;
}
