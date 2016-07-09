#pragma once

#include <Python.h>
#include <complex>
#include <vector>

enum class Type
{
    Float,
    FloatBuffer,
    ComplexBuffer,
    Unsupported
};

template <typename T>
struct Buffer
{
    Buffer(int samples, int channels, int frames, T* data)
      : data(data)
      , samples(samples)
      , channels(channels)
      , frames(frames)
    {
    }
    T* data;
    int samples, channels, frames;
};
typedef Buffer<float> FloatBuffer;
typedef Buffer<std::complex<float>> ComplexBuffer;

class Object
{
  public:
    Object(float value);
    Object(FloatBuffer& ptr);
    Object(ComplexBuffer& value);
    Object(PyObject* obj);
    void destroy();

    Type type();
    PyObject* getPyObject();
    float& getFloat();
    FloatBuffer& getFloatBuffer();
    ComplexBuffer& getComplexBuffer();

    void send();
    void recv();

  private:
    PyObject* _obj;
    void* _value;
    Type _type;
};
