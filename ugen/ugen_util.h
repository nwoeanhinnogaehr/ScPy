#pragma once

#include <SC_PlugIn.h>
#include <string>

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
