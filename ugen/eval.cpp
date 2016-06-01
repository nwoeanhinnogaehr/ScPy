#include "eval.h"
#include <Python.h>
#include <iostream>

Evaluator::Evaluator()
{
    setenv("PYTHONPATH", ABS_SOURCE_PATH "/../py", 1);
    Py_Initialize();
}

Evaluator::~Evaluator()
{
    Py_Finalize();
}

void
Evaluator::eval(const char* code)
{
    PyRun_SimpleString(code);
}
