#include "eval.h"
#include "sc_objs.h"
#include "ugen_util.h"
#include <FFT_UGens.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

InterfaceTable* ft;
static Evaluator eval;

struct Py : public Unit
{
    string code;
    PyObject* obj;
    int doneAction;
    vector<Object*> objs;
};

extern "C" {
void Py_Ctor(Py* unit);
void Py_Dtor(Py* unit);
void Py_Next(Py* unit, int numSamples);
}

void
done(Py* unit)
{
    unit->mDone = true;
    DoneAction(unit->doneAction, unit);
    SETCALC(*ClearUnitOutputs);
}

bool
checkError(Py* unit)
{
    if (eval.checkError()) {
        eval.printError();
        unit->doneAction = 2;
        done(unit);
        return true;
    }
    return false;
}

enum Type
{
    Float,
    FloatBuffer,
    ComplexBuffer,
    Array,
    UGen,
    Unsupported
};

Type
parseType(string& type)
{
    if (type == "Integer" || type == "Float")
        return Type::Float;
    if (type == "Buffer")
        return Type::FloatBuffer;
    if (type == "FFT")
        return Type::ComplexBuffer;
    if (type == "Array")
        return Type::Array;
    if (type == "UGen")
        return Type::UGen;
    return Type::Unsupported;
}

Object*
readObject(Py* unit, int& idx)
{
    string typeStr = readString(unit, idx);
    Type type = parseType(typeStr);

    switch (type) {
        case Type::FloatBuffer: {
            uint32 bufNum = readAtom<uint32>(unit, idx);
            SndBuf* buf = getSndBuf(unit, bufNum);
            FloatBufferObject* obj = new FloatBufferObject(
              buf->samples, buf->channels, buf->frames, buf->data);
            return obj;
        }
        case Type::ComplexBuffer: {
            uint32 bufNum = readAtom<uint32>(unit, idx);
            SndBuf* buf = getSndBuf(unit, bufNum);
            SCComplexBuf* complexBuf = ToComplexApx(buf);
            ComplexBufferObject* obj = new ComplexBufferObject(
              buf->samples / 2, buf->channels, buf->frames / 2,
              reinterpret_cast<std::complex<float>*>(complexBuf));
            return obj;
        }
        case Type::Array: {
            int numItems = readAtom<int>(unit, idx);
            vector<Object*> arrayItems;
            for (int j = 0; j < numItems; j++) {
                Object* obj = readObject(unit, idx);
                if (!obj)
                    return nullptr;
                arrayItems.push_back(obj);
            }
            ArrayObject* obj = new ArrayObject(arrayItems);
            return obj;
        }
        case Type::Float: {
            float value = readAtom<float>(unit, idx);
            ConstObject* obj = new ConstObject(value);
            return obj;
        }
        case Type::UGen: {
            float* ptr = IN(idx++);
            ControlUGenObject* obj = new ControlUGenObject(ptr);
            return obj;
        }
        case Type::Unsupported:
            cout << "Argument has unsupported type '" << typeStr << "'" << endl;
            done(unit);
            return nullptr;
    }
}

void
Py_Ctor(Py* unit)
{
    cout << "Py_Ctor" << endl;
    new (unit) Py;

    int idx = 0;
    unit->doneAction = readAtom<int>(unit, idx);
    unit->code = readString(unit, idx);
    unit->obj = eval.compile(unit->code);
    if (checkError(unit))
        return;

    int numArgs = readAtom<int>(unit, idx);
    for (int i = 0; i < numArgs; i++) {
        string name = readString(unit, idx);
        Object* obj = readObject(unit, idx);
        if (!obj)
            return;
        unit->objs.emplace_back(obj);
        eval.defineVariable(name, obj);
    }

    SETCALC(Py_Next);
}

void
Py_Dtor(Py* unit)
{
    cout << "Py_Dtor" << endl;
    for (Object* o : unit->objs) {
        o->destroy();
        delete o;
    }
    unit->~Py();
}

bool
bufferReady(Py* unit, int& idx)
{
    string typeStr = readString(unit, idx);
    Type type = parseType(typeStr);
    switch (type) {
        case Type::FloatBuffer:
        case Type::ComplexBuffer: {
            float fBufNum = readAtom<float>(unit, idx);
            return fBufNum >= 0.0;
        }
        case Array: {
            int numItems = readAtom<int>(unit, idx);
            for (int j = 0; j < numItems; j++) {
                if (!bufferReady(unit, idx)) {
                    return false;
                }
            }
            return true;
        }
        case Type::UGen:
        case Type::Float:
            idx++;
            return true;
        case Type::Unsupported:
            return false;
    }
}

void
Py_Next(Py* unit, int)
{
    int idx = 0;
    readAtom<int>(unit, idx); // doneAction
    readString(unit, idx);    // code
    int numArgs = readAtom<int>(unit, idx);
    for (int i = 0; i < numArgs; i++) {
        readString(unit, idx); // name
        if (!bufferReady(unit, idx))
            return;
    }

    for (Object* o : unit->objs)
        o->send();        // data to python
    eval.eval(unit->obj); // call python
    for (Object* o : unit->objs)
        o->recv(); // data from python

    if (checkError(unit))
        return;
    if (unit->doneAction)
        done(unit);
}

PluginLoad(Py)
{
    ft = inTable;
    DefineDtorUnit(Py);
}
