#pragma once

#include "object.h"
#include <Python.h>
#include <string>

class Evaluator
{
  public:
    Evaluator();
    ~Evaluator();

    PyObject* eval(PyObject* obj);
    PyObject* compile(const std::string& code);
    bool checkError();
    void printError();
    void defineVariable(const std::string& name, Object obj);

  private:
    PyObject *_globals, *_locals;
    PyObject* _flusher;
};
