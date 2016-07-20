#pragma once

#include <Python.h>

class Object
{
  public:
    Object();
    virtual ~Object();
    virtual void send() { };
    virtual void recv() { };
    virtual void destroy();
    virtual PyObject* pyObject();

  protected:
    PyObject* _obj;
};
