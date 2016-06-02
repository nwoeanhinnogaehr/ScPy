#include "eval.h"
#include <iostream>

Evaluator::Evaluator()
{
    setenv("PYTHONPATH", ABS_SOURCE_PATH "/../py", 1);
    Py_Initialize();
    PyObject* main = PyImport_AddModule("__main__");
    _globals = PyModule_GetDict(main);
    _locals = PyDict_New();
    PyObject* apiModule = PyImport_ImportModule("api");
    PyObject* apiAttrib = PyObject_Dir(apiModule);
    Py_ssize_t numAttrs = PyList_Size(apiAttrib);
    for (Py_ssize_t i = 0; i < numAttrs; i++) {
        PyObject* val = PyList_GetItem(apiAttrib, i);
        PyModule_AddObject(main, PyUnicode_AsUTF8(val), PyObject_GetAttr(apiModule, val));
    }
}

Evaluator::~Evaluator()
{
    Py_Finalize();
}

PyObject*
Evaluator::compile(const char* code)
{
    PyObject* obj =
      Py_CompileStringExFlags(code, "sc-anon", Py_single_input, nullptr, 0);
    return obj;
}

PyObject*
Evaluator::eval(PyObject* code)
{
    PyObject* res = PyEval_EvalCode(code, _globals, _locals);
    return res;
}
