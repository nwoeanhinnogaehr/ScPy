#pragma once

#include <Python.h>

class Evaluator
{
  public:
    Evaluator();
    ~Evaluator();

    PyObject* eval(PyObject* obj);
    PyObject* compile(const char* code);

  private:
    PyObject *_globals, *_locals;
};
