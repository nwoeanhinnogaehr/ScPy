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
    int codeSize = (int)ZIN0(0);
    char code[codeSize + 1] = { 0 };
    for (int i = 0; i < codeSize; i++) {
        code[i] = (char)ZIN0(1 + i);
    }

    unit->eval->eval(code);
}

PluginLoad(FSM)
{
    // InterfaceTable *inTable implicitly given as argument to the load function
    ft = inTable; // store pointer to InterfaceTable

    DefineDtorUnit(FSM);
}
