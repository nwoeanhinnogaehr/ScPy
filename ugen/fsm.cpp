#include "eval.h"
#include "ugen_util.h"
#include <FFT_UGens.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

InterfaceTable* ft;
static Evaluator eval;

struct FSM : public Unit
{
    string code;
    PyObject* obj;
    int doneAction;
    vector<Object> objs;
};

extern "C" {
void FSM_Ctor(FSM* unit);
void FSM_Dtor(FSM* unit);
void FSM_Next(FSM* unit, int numSamples);
}

void
done(FSM* unit)
{
    unit->mDone = true;
    DoneAction(unit->doneAction, unit);
    SETCALC(*ClearUnitOutputs);
}

bool
checkError(FSM* unit)
{
    if (eval.checkError()) {
        eval.printError();
        unit->doneAction = 2;
        done(unit);
        return true;
    }
    return false;
}

Type
parseType(string& type)
{
    if (type == "Integer" || type == "Float")
        return Type::Float;
    if (type == "Buffer")
        return Type::FloatBuffer;
    if (type == "FFT")
        return Type::ComplexBuffer;
    return Type::Unsupported;
}

void
FSM_Ctor(FSM* unit)
{
    cout << "FSM_Ctor" << endl;
    new (unit) FSM;

    int idx = 0;
    unit->doneAction = readAtom<int>(unit, idx);
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
                Object obj(val);
                unit->objs.emplace_back(obj);
                eval.defineVariable(name, obj);
                break;
            }
            case Type::FloatBuffer: {
                uint32 bufNum = readAtom<uint32>(unit, idx);
                FloatBuffer buf = getFloatBuffer(unit, bufNum);
                Object obj(buf);
                unit->objs.emplace_back(obj);
                eval.defineVariable(name, obj);
                break;
            }
            case Type::ComplexBuffer: {
                uint32 bufNum = readAtom<uint32>(unit, idx);
                ComplexBuffer buf = getComplexBuffer(unit, bufNum);
                Object obj(buf);
                unit->objs.emplace_back(obj);
                eval.defineVariable(name, obj);
                break;
            }
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
    for (Object &o : unit->objs) o.destroy();
    unit->~FSM();
}

void
FSM_Next(FSM* unit, int)
{
    int idx = 0;
    readAtom<int>(unit, idx); // doneAction
    readString(unit, idx); // code
    int numArgs = readAtom<int>(unit, idx);
    for (int i = 0; i < numArgs; i++) {
        readString(unit, idx); // name
        string typeStr = readString(unit, idx);
        Type type = parseType(typeStr);
        switch (type) {
            case Type::FloatBuffer:
            case Type::ComplexBuffer: {
                float fBufNum = readAtom<float>(unit, idx);
                if (fBufNum < 0.0)
                    return;
            }
            case Type::Float:
            case Type::Unsupported:
                break;
        }
    }

    for (Object &o : unit->objs) o.send(); // data to python
    eval.eval(unit->obj); // call python
    for (Object &o : unit->objs) o.recv(); // data from python

    if (checkError(unit))
        return;
    if (unit->doneAction)
        done(unit);
}

PluginLoad(FSM)
{
    ft = inTable;
    DefineDtorUnit(FSM);
}
