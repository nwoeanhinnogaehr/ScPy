#pragma once

#include "object.h"
#include <complex>
#include <memory>
#include <vector>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL FSM_ARRAY_API
#include <numpy/arrayobject.h>

using namespace std;

template <typename T>
class BufferObject : public Object
{
  public:
    BufferObject(int samples, int channels, int frames, T* data, int type)
      : _data(data)
      , _samples(samples)
      , _channels(channels)
      , _frames(frames)
      , _type(type)
    {
        if (channels > 1) {
            long dims[2] = { (long)channels, (long)frames };
            _obj = PyArray_SimpleNew(2, dims, type);
        } else {
            long dims[1] = { (long)frames };
            _obj = PyArray_SimpleNew(1, dims, type);
        }
    }

  protected:
    T* _data;
    int _samples, _channels, _frames, _type;
};

class FloatBufferObject : public BufferObject<float>
{
  public:
    FloatBufferObject(int samples, int channels, int frames, float* data);
    virtual ~FloatBufferObject();
    virtual void send();
    virtual void recv();
};

class ComplexBufferObject : public BufferObject<std::complex<float>>
{
  public:
    ComplexBufferObject(int samples, int channels, int frames,
                        std::complex<float>* data);
    virtual ~ComplexBufferObject();
    virtual void send();
    virtual void recv();
};

class ArrayObject : public Object
{
  public:
    ArrayObject(vector<Object*> objects);
    virtual void send();
    virtual void recv();

  protected:
    vector<Object*> _objs;
};

class ConstObject : public Object
{
  public:
    ConstObject(float value);
};
