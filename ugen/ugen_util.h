#pragma once

#include "object.h"
#include <FFT_UGens.h>
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
