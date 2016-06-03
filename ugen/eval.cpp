#include "eval.h"
#include <iostream>

void
importUnqualified(PyObject* main, const char* name)
{
    PyObject* module = PyImport_ImportModule(name);
    PyObject* attrib = PyObject_Dir(module);
    Py_ssize_t numAttrs = PyList_Size(attrib);
    for (Py_ssize_t i = 0; i < numAttrs; i++) {
        PyObject* val = PyList_GetItem(attrib, i);
        PyModule_AddObject(main, PyUnicode_AsUTF8(val),
                           PyObject_GetAttr(module, val));
    }
}

void
importQualified(PyObject* main, const char* name, const char* as)
{
    PyObject* module = PyImport_ImportModule(name);
    PyModule_AddObject(main, as, module);
}

Evaluator::Evaluator()
{
    setenv("PYTHONPATH", ABS_SOURCE_PATH "/../py", 1);
    Py_Initialize();
    PyObject* main = PyImport_AddModule("__main__");
    _globals = PyModule_GetDict(main);
    _locals = PyDict_New();

    importUnqualified(main, "api");
    importQualified(main, "numpy", "np");
}

Evaluator::~Evaluator()
{
    Py_Finalize();
}

PyObject*
Evaluator::compile(const char* code)
{
    PyObject* obj =
      Py_CompileStringExFlags(code, "sc-anon", Py_file_input, nullptr, 0);
    return obj;
}

PyObject*
Evaluator::eval(PyObject* code)
{
    PyObject* res = PyEval_EvalCode(code, _globals, _locals);
    return res;
}
