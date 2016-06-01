#include "eval.h"
#include <SC_PlugIn.h>
#include <iostream>

using namespace std;

static InterfaceTable* ft;

struct FSM : public Unit
{
    Evaluator* eval;
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

    SETCALC(FSM_Next);
    FSM_Next(unit, 1);
}

void
FSM_Dtor(FSM* unit)
{
    cout << "FSM_Dtor" << endl;

    delete unit->eval;
}

void
FSM_Next(FSM* unit, int numSamples)
{
    // the code to evaluate is passed in by setting the first argument to the
    // length of the string and that many subsequent arguments to ASCII values
    // represented as floating point numbers
    // ...
    // apparently this is the best way to do it.
    int codeSize = (int)ZIN0(0);
    char code[codeSize + 1];
    code[codeSize] = 0; // ensure null terminated
    for (int i = 0; i < codeSize; i++) {
        code[i] = (char)ZIN0(1 + i);
    }

    unit->eval->eval(code);
}


PluginLoad(FSM)
{
    ft = inTable;
    DefineDtorUnit(FSM);
}
