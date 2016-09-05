#include "eval.h"
#include <iostream>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL SCPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <sstream>
#include <vector>

using namespace std;

void
importUnqualified(PyObject* main, const char* name)
{
    PyObject* module = PyImport_ImportModule(name);
    if (!module) {
        PyErr_Print();
        return;
    }
    PyObject* attrib = PyObject_Dir(module);
    if (!attrib) {
        PyErr_Print();
        return;
    }
    Py_ssize_t numAttrs = PyList_Size(attrib);
    for (Py_ssize_t i = 0; i < numAttrs; i++) {
        PyObject* val = PyList_GetItem(attrib, i);
        if (!val) {
            PyErr_Print();
            return;
        }
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
    cout << "Evaluator()" << endl;
    setenv("PYTHONPATH", ABS_SOURCE_PATH "/../py", 1);
    Py_Initialize();
    _import_array();
    PyObject* main = PyImport_AddModule("__main__");
    _globals = PyModule_GetDict(main);
    _locals = PyDict_New();

    //importUnqualified(main, "api");
    //importUnqualified(main, "numpy");

    _flusher = compile("import sys\nsys.stdout.flush()");
}

Evaluator::~Evaluator()
{
    cout << "~Evaluator()" << endl;
    Py_Finalize();
}

PyObject*
Evaluator::compile(const string& code)
{
    // preprocess
    istringstream ss(code);
    ostringstream out;
    vector<string> lines;
    string line;
    size_t minIndent = 9999999;
    while (getline(ss, line)) {
        lines.push_back(line);
        size_t i = 0;
        while (line[i++] == ' ')
            ;
        if (i < line.length())
            minIndent = min(minIndent, i - 1);
    }
    out << "import sys\nsys.setdlopenflags(9)\nimport api\nimport numpy\nnumpy.seterr(all='ignore')\n";
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].length() > minIndent)
            out << lines[i].substr(minIndent) << "\n";
        else
            out << lines[i] << "\n";
    }
    out << "globals().update(locals())\n";
    // out << "import gc\ngc.collect()\n";

    PyObject* obj = Py_CompileStringExFlags(out.str().c_str(), "sc-anon",
                                            Py_file_input, nullptr, 0);
    return obj;
}

PyObject*
Evaluator::eval(PyObject* code)
{
    PyObject* res = PyEval_EvalCode(code, _globals, _locals);
    if (res)
        PyEval_EvalCode(_flusher, _globals, _locals); // flush stdout
    return res;
}

bool
Evaluator::checkError()
{
    return PyErr_Occurred() != nullptr;
}

void
Evaluator::printError()
{
    PyErr_Print();
    cout << flush;
}

void
Evaluator::defineVariable(const std::string& name, Object* obj)
{
    PyDict_SetItemString(_locals, name.c_str(), obj->pyObject());
}
