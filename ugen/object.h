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
    FloatArray(int samples, int channels, int frames, float* data)
      : data(data)
      , samples(samples)
      , channels(channels)
      , frames(frames)
    {
    }
    float* data;
    int samples, channels, frames;
};
struct ComplexArray
{
    ComplexArray(int samples, int channels, int frames,
                 std::complex<float>* data)
      : data(data)
      , samples(samples)
      , channels(channels)
      , frames(frames)
    {
    }
    std::complex<float>* data;
    int samples, channels, frames;
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
    float& getFloat();
    FloatArray& getFloatArray();
    ComplexArray& getComplexArray();

  private:
    PyObject* _obj;
    void* _value;
    Type _type;
};
