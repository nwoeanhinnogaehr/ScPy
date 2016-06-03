#include "eval.h"
#include <Python.h>
#include <SC_PlugIn.h>
#include <iostream>

using namespace std;

static InterfaceTable* ft;
static Evaluator eval;

struct FSM : public Unit
{
    char* code;
    PyObject* obj;
};

extern "C" {
void FSM_Ctor(FSM* unit);
void FSM_Dtor(FSM* unit);
void FSM_Next(FSM* unit, int numSamples);
void FSM_NextVoid(FSM* unit, int numSamples);
}

void
done(FSM* unit)
{
    unit->mDone = true;
    DoneAction(13, unit);
    SETCALC(FSM_NextVoid);
}

bool
checkError(FSM *unit)
{
    if (eval.checkError()) {
        eval.printError();
        done(unit);
        return true;
    }
    return false;
}

void
FSM_Ctor(FSM* unit)
{
    cout << "FSM_Ctor" << endl;

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

    unit->obj = eval.compile(unit->code);
    if (checkError(unit))
        return;

    SETCALC(FSM_Next);
}

void
FSM_Dtor(FSM* unit)
{
    cout << "FSM_Dtor" << endl;

    delete[] unit->code;
}

void
FSM_Next(FSM* unit, int)
{
    eval.eval(unit->obj);
    if (checkError(unit))
        return;
    done(unit); // for now, only run once
}

void
FSM_NextVoid(FSM*, int)
{
}

PluginLoad(FSM)
{
    ft = inTable;
    DefineDtorUnit(FSM);
}
