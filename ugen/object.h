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

struct FloatArray
{
    FloatArray(int channels, int frames, std::vector<float> data)
      : data(data)
      , channels(channels)
      , frames(frames)
    {
    }
    std::vector<float> data;
    int channels, frames;
};
struct ComplexArray
{
    ComplexArray(int channels, int frames,
                 std::vector<std::complex<float>> data)
      : data(data)
      , channels(channels)
      , frames(frames)
    {
    }
    std::vector<std::complex<float>> data;
    int channels, frames;
};

class Object
{
  public:
    Object(float value);
    Object(FloatArray& ptr);
    Object(ComplexArray& value);
    Object(PyObject* obj);
    ~Object();

    Type type();
    PyObject* getPyObject();
    float getFloat();
    FloatArray& getFloatArray();
    ComplexArray& getComplexArray();

  private:
    PyObject* _obj;
    void* _value;
    Type _type;
};
