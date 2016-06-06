#include "object.h"

Object::Object(float value)
{
    _value = new float(value);
    _type = Type::Float;
    _obj = Py_BuildValue("f", value);
}

Object::Object(std::vector<float> ptr)
{
}

Object::Object(std::vector<std::complex<float>> value)
{
}

Object::Object(PyObject* obj)
{
}

Object::~Object()
{
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

float
Object::getFloat()
{
    return *(float*)_value;
}

std::vector<float>
Object::getFloatArray()
{
}

std::vector<std::complex<float>>
Object::getComplexArray()
{
}
