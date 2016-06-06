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

template <typename T>
T
readAtom(FSM* unit, int& idx)
{
    return (T)ZIN0(idx++);
}

string
readString(FSM* unit, int& idx)
{
    int length = readAtom<int>(unit, idx);
    string s;
    for (int i = 0; i < length; i++) {
        s += readAtom<char>(unit, idx);
    }
    return s;
}

Type
parseType(string& type)
{
    if (type == "Integer" || type == "Float")
        return Type::Float;
    if (type == "Buffer")
        return Type::FloatArray;
    if (type == "FFT")
        return Type::ComplexArray;
    return Type::Unsupported;
}

void
FSM_Ctor(FSM* unit)
{
    cout << "FSM_Ctor" << endl;
    new (unit) FSM;

    int idx = 0;
    unit->code = readString(unit, idx);
    unit->obj = eval.compile(unit->code);
    if (checkError(unit))
        return;

    int numArgs = readAtom<int>(unit, idx);
    for (int i = 0; i < numArgs; i++) {
        string name = readString(unit, idx);
        string typeStr = readString(unit, idx);
        Type type = parseType(typeStr);
        switch (type) {
            case Type::Float: {
                float val = readAtom<float>(unit, idx);
                eval.defineGlobal(name, Object(val));
                break;
            }
            case Type::FloatArray:
                break;
            case Type::ComplexArray:
                break;
            case Type::Unsupported:
                cout << "Argument '" << name << "' has unsupported type '"
                     << typeStr << "'" << endl;
                done(unit);
                return;
        }
        cout << name << " :: " << typeStr << endl;
    }

    SETCALC(FSM_Next);
}

void
FSM_Dtor(FSM* unit)
{
    cout << "FSM_Dtor" << endl;
    unit->~FSM();
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
