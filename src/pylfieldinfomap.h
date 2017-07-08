/*
 * pylfieldinfomap.h
 *
 *
 * This file is part of PyLidar
 * Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PYLFIELDINFOMAP_H
#define PYLFIELDINFOMAP_H

#include <Python.h>
#include "numpy/arrayobject.h"
#include <string>
#include <string.h> // for memcpy
#include <map>

#define DO_INT64_READ(tempVar) memcpy(&tempVar, (char*)pRow + info.nOffset, sizeof(tempVar)); \
            nRetVal = (npy_int64)tempVar; 

#define DO_FLOAT64_READ(tempVar) memcpy(&tempVar, (char*)pRow + info.nOffset, sizeof(tempVar)); \
            dRetVal = (double)tempVar; 

namespace pylidar
{

// to help us remember info for each field without having to look it up each time
// used by CFieldInfoMap
typedef struct {
    char cKind;
    int nOffset;
    int nSize;
} SFieldInfo;

// Class to make extracting values from numpy arrays much easier
class CFieldInfoMap : public std::map<std::string, SFieldInfo>
{
public:
    CFieldInfoMap(PyArrayObject *pArray) 
    {
        if( (pArray == NULL ) || ((PyObject*)pArray == Py_None) )
            return;

        if( !PyArray_Check(pArray) )
        {
            fprintf(stderr, "CFieldInfoMap not being passed an array");
            return;
        }

        // populate the map from the fields in the array
        SFieldInfo info;
        PyArray_Descr *pDescr = PyArray_DESCR(pArray);
        PyObject *pKeys = PyDict_Keys(pDescr->fields);
        for( Py_ssize_t i = 0; i < PyList_Size(pKeys); i++)
        {
            PyObject *pKey = PyList_GetItem(pKeys, i);
#if PY_MAJOR_VERSION >= 3
            PyObject *bytesKey = PyUnicode_AsEncodedString(pKey, NULL, NULL);
            char *pszElementName = PyBytes_AsString(bytesKey);
#else
            char *pszElementName = PyString_AsString(pKey);
#endif

            pylidar_getFieldDescr(pArray, pszElementName, &info.nOffset, &info.cKind, &info.nSize, NULL);
            insert( std::pair<std::string, SFieldInfo>(pszElementName, info) );

#if PY_MAJOR_VERSION >= 3
            Py_DECREF(bytesKey);
#endif
        }
        Py_DECREF(pKeys);
    }

    // get a field as int64
    npy_int64 getIntValue(std::string sName, void *pRow, npy_int64 nDefault=0)
    {
        npy_char nCharVal;
        npy_bool nBoolVal;
        npy_byte nByteVal;
        npy_ubyte nUByteVal;
        npy_short nShortVal;
        npy_ushort nUShortVal;
        npy_int nIntVal;
        npy_uint nUIntVal;
        npy_long nLongVal;
        npy_ulong nULongVal;
        npy_float fFloatVal;
        npy_double fDoubleVal;
        npy_int64 nRetVal=nDefault;

        iterator it = find(sName);
        if( it == end() )
        {
            return nDefault;
        }
        SFieldInfo info = it->second;
        if( ( info.cKind == 'b' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nBoolVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nByteVal);
        }
        else if ( ( info.cKind == 'S' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nCharVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nUByteVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 2 ) )
        {
            DO_INT64_READ(nShortVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 2 ) )
        {
            DO_INT64_READ(nUShortVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 4 ) )
        {
            DO_INT64_READ(nIntVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 4 ) )
        {
            DO_INT64_READ(nUIntVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 8 ) )
        {
            DO_INT64_READ(nLongVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 8 ) )
        {
            DO_INT64_READ(nULongVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 4 ) )
        {
            DO_INT64_READ(fFloatVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 8 ) )
        {
            DO_INT64_READ(fDoubleVal);
        }
        return nRetVal;        
    }

    // get a field as double
    double getDoubleValue(std::string sName, void *pRow, double dDefault=0.0)
    {
        npy_char nCharVal;
        npy_bool nBoolVal;
        npy_byte nByteVal;
        npy_ubyte nUByteVal;
        npy_short nShortVal;
        npy_ushort nUShortVal;
        npy_int nIntVal;
        npy_uint nUIntVal;
        npy_long nLongVal;
        npy_ulong nULongVal;
        npy_float fFloatVal;
        npy_double fDoubleVal;
        double dRetVal=dDefault;

        iterator it = find(sName);
        if( it == end() )
        {
            return dDefault;
        }

        SFieldInfo info = it->second;
        if( ( info.cKind == 'b' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nBoolVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nByteVal);
        }
        else if ( ( info.cKind == 'S' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nCharVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nUByteVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 2 ) )
        {
            DO_FLOAT64_READ(nShortVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 2 ) )
        {
            DO_FLOAT64_READ(nUShortVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 4 ) )
        {
            DO_FLOAT64_READ(nIntVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 4 ) )
        {
            DO_FLOAT64_READ(nUIntVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 8 ) )
        {
            DO_FLOAT64_READ(nLongVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 8 ) )
        {
            DO_FLOAT64_READ(nULongVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 4 ) )
        {
            DO_FLOAT64_READ(fFloatVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 8 ) )
        {
            DO_FLOAT64_READ(fDoubleVal);
        }
        return dRetVal;        
    }

};

} //namespace pylidar

#endif //PYLFIELDINFOMAP_H

