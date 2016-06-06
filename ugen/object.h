#pragma once

#include <Python.h>
#include <complex>
#include <vector>

enum class Type
{
    Float,
    FloatArray,
    ComplexArray,
    Unsupported
};

class Object
{
  public:
    Object(float value);
    Object(std::vector<float> ptr);
    Object(std::vector<std::complex<float>> value);
    Object(PyObject* obj);
    ~Object();

    Type type();
    PyObject* getPyObject();
    float getFloat();
    std::vector<float> getFloatArray();
    std::vector<std::complex<float>> getComplexArray();

  private:
    PyObject* _obj;
    void* _value;
    Type _type;
};
