#include "object.h"

Object::Object()
{
}

Object::~Object()
{
}

void
Object::destroy()
{
    Py_DecRef(_obj);
}

PyObject*
Object::pyObject()
{
    return _obj;
}
