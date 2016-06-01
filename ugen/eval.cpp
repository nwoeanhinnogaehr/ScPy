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
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* globals = PyModule_GetDict(main);
    PyObject* locals = PyDict_New();
    PyObject* obj = PyRun_String(code, Py_eval_input, globals, locals);
    if (!obj) {
        PyErr_Print();
        return;
    }
    PyObject_Print(obj, stdout, 0);
}
