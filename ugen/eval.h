#pragma once

#include <Python.h>

class Evaluator
{
  public:
    Evaluator();
    ~Evaluator();

    PyObject* eval(PyObject* obj);
    PyObject* compile(const char* code);
    bool checkError();
    void printError();

  private:
    PyObject *_globals, *_locals;
    PyObject *_flusher;
};
