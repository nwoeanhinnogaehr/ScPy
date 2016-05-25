#include <iostream>
#include <SC_PlugIn.h>

using namespace std;

static InterfaceTable *ft;

struct FSM : public Unit {
};

extern "C" {
    void FSM_Ctor(FSM *unit);
    void FSM_Dtor(FSM *unit);
    void FSM_Next(FSM *unit, int numSamples);
}

void FSM_Ctor(FSM *unit) {
    cout << "FSM_Ctor" << endl;
    SETCALC(FSM_Next);
    FSM_Next(unit, 1);
}

void FSM_Dtor(FSM *unit) {
    cout << "FSM_Dtor" << endl;
}

void FSM_Next(FSM *unit, int numSamples) {
    cout << "FSM_Next" << endl;

    int codeSize = (int)ZIN0(0);
    char code[codeSize + 1] = { 0 };
    for (int i = 0; i < codeSize; i++) {
        code[i] = (char)ZIN0(1 + i);
    }
    cout << code << endl;
}

PluginLoad(FSM) {
    // InterfaceTable *inTable implicitly given as argument to the load function
    ft = inTable; // store pointer to InterfaceTable

    DefineDtorUnit(FSM);
}
