#pragma once

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

  private:
    PyObject *_globals, *_locals;
    PyObject *_flusher;
};
