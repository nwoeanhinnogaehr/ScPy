#pragma once

#include "object.h"
#include <SC_PlugIn.h>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
T
readAtom(Unit* unit, int& idx)
{
    return (T)ZIN0(idx++);
}

std::string
readString(Unit* unit, int& idx)
{
    int length = readAtom<int>(unit, idx);
    std::string s;
    for (int i = 0; i < length; i++) {
        s += readAtom<char>(unit, idx);
    }
    return s;
}

SndBuf*
getSndBuf(Unit* unit, uint32 bufNum)
{
    World* world = unit->mWorld;
    SndBuf* buf;
    if (bufNum >= world->mNumSndBufs) {
        int localBufNum = bufNum - world->mNumSndBufs;
        Graph* parent = unit->mParent;
        if (localBufNum <= parent->localBufNum) {
            buf = parent->mLocalSndBufs + localBufNum;
        } else {
            buf = world->mSndBufs;
        }
    } else {
        buf = world->mSndBufs + bufNum;
    }
    return buf;
}

FloatArray
getFloatBuffer(Unit* unit, uint32 bufNum)
{
    SndBuf* buf = getSndBuf(unit, bufNum);
    LOCK_SNDBUF(buf);
    FloatArray out(buf->channels, buf->frames,
                   std::vector<float>(buf->data, buf->data + buf->samples));
    return out;
}

void
setFloatBuffer(Unit* unit, uint32 bufNum, FloatArray& arr)
{
    // TODO
}
