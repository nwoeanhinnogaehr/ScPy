#include "eval.h"
#include <SC_PlugIn.h>
#include <iostream>
#include <Python.h>

using namespace std;

static InterfaceTable* ft;

struct FSM : public Unit
{
    Evaluator* eval;
    char* code;
    PyObject* obj;
};

extern "C" {
void FSM_Ctor(FSM* unit);
void FSM_Dtor(FSM* unit);
void FSM_Next(FSM* unit, int numSamples);
}

void
FSM_Ctor(FSM* unit)
{
    cout << "FSM_Ctor" << endl;

    unit->eval = new Evaluator;

    // the code to evaluate is passed in by setting the first argument to the
    // length of the string and that many subsequent arguments to ASCII values
    // represented as floating point numbers
    // ...
    // apparently this is the best way to do it.
    int codeSize = (int)ZIN0(0);
    unit->code = new char[codeSize + 1];
    unit->code[codeSize] = 0; // ensure null terminated
    for (int i = 0; i < codeSize; i++) {
        unit->code[i] = (char)ZIN0(1 + i);
    }

    unit->obj = unit->eval->compile(unit->code);

    SETCALC(FSM_Next);
}

void
FSM_Dtor(FSM* unit)
{
    cout << "FSM_Dtor" << endl;

    delete unit->eval;
    delete[] unit->code;
}

void
FSM_Next(FSM* unit, int numSamples)
{
    unit->eval->eval(unit->obj);
}

PluginLoad(FSM)
{
    ft = inTable;
    DefineDtorUnit(FSM);
}
