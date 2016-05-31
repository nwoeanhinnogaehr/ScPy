#include "eval.h"
#include <Python.h>
#include <iostream>

Evaluator::Evaluator()
{
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
