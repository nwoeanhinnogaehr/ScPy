#include "eval.h"
#include <Python.h>
#include <SC_PlugIn.h>
#include <iostream>
#include <string>

using namespace std;

static InterfaceTable* ft;
static Evaluator eval;

struct FSM : public Unit
{
    string code;
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
checkError(FSM* unit)
{
    if (eval.checkError()) {
        eval.printError();
        done(unit);
        return true;
    }
    return false;
}

string
readString(FSM* unit, int idx)
{
    string s((size_t)ZIN0(idx), 0);
    for (size_t i = 0; i < s.length(); i++) {
        s[i] = (char)ZIN0(1 + i);
    }
    return s;
}

void
FSM_Ctor(FSM* unit)
{
    cout << "FSM_Ctor" << endl;

    unit->code = readString(unit, 0);
    unit->obj = eval.compile(unit->code);
    if (checkError(unit))
        return;

    SETCALC(FSM_Next);
}

void
FSM_Dtor(FSM*)
{
    cout << "FSM_Dtor" << endl;
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
