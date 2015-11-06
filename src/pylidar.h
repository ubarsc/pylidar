/*
 * pylidar.h
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

#ifndef PYLIDAR_H
#define PYLIDAR_H

#include <Python.h>
#include "numpy/arrayobject.h"

#ifdef __cplusplus
extern "C" {
#endif

/* mark all exported classes/functions with DllExport to have
 them exported by Visual Studio */
#ifndef DllExport
    #ifdef _MSC_VER
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport
    #endif
#endif

/* MSVC 2008 uses different names.... */
#ifdef _MSC_VER
    #if _MSC_VER >= 1600
        #include <stdint.h>
    #else        
        typedef __int8              int8_t;
        typedef __int16             int16_t;
        typedef __int32             int32_t;
        typedef __int64             int64_t;
        typedef unsigned __int8     uint8_t;
        typedef unsigned __int16    uint16_t;
        typedef unsigned __int32    uint32_t;
        typedef unsigned __int64    uint64_t;
    #endif
#else
    #include <stdint.h>
#endif

/* call first - preferably in the init() of your module
 this sets up the connection to numpy */
#if PY_MAJOR_VERSION >= 3
PyObject *pylidar_init();
#else
DllExport void pylidar_init();
#endif
/* print error - used internally */
DllExport void pylidar_error(char *errstr, ...);

/* Helper function to get information about a named field within an array
 pass null for params you not interested in */
DllExport int pylidar_getFieldDescr(PyObject *pArray, const char *pszName, int *pnOffset, char *pcKind, int *pnSize, int *pnLength);

/* return a field as a int64_t array. Caller to delete */
DllExport int64_t *pylidar_getFieldAsInt64(PyObject *pArray, const char *pszName);

/* return a field as a double array. Caller to delete */
DllExport double *pylidar_getFieldAsFloat64(PyObject *pArray, const char *pszName);


/* structure for defining a numpy structured array from a C one
 create using CREATE_FIELD_DEFN below */
typedef struct
{
    const char *pszName;
    char cKind; // 'i' for signed int, 'u' for unsigned int, 'f' for float
    int nSize;
    int nOffset;
    int nStructTotalSize;
} SpylidarFieldDefn;

#define CREATE_FIELD_DEFN(STRUCT, FIELD, KIND) \
    {#FIELD, KIND, sizeof(((STRUCT*)0)->FIELD), offsetof(STRUCT, FIELD), sizeof(STRUCT)}

/* 
Here is an example of use:
//Say you had a structure like this:
typedef struct {
    double x,
    double y,
    int count
} SMyStruct;

//Create an array of structures defining the fields like this:
static SpylidarFieldDefn fields[] = {
    CREATE_FIELD_DEFN(SMyStruct, x, 'f'),
    CREATE_FIELD_DEFN(SMyStruct, y, 'f'),
    CREATE_FIELD_DEFN(SMyStruct, count, 'i'),
    {NULL} // Sentinel 
};

// Kind is one of the following (see http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)
'b'     boolean
'i'     (signed) integer
'u'     unsigned integer
'f'     floating-point
'c'     complex-floating point
'O'     (Python) objects
'S', 'a'    (byte-)string
'U'     Unicode
'V'     raw data (void)

*/
/* Wrap an existing C array of structures and return as a numpy array */
/* Python will free data when finished */
DllExport PyObject *pylidar_structArrayToNumpy(void *pStructArray, npy_intp nElems, SpylidarFieldDefn *pDefn);

#ifdef __cplusplus
} // extern C

#include <new>
#include <algorithm>

template <class T>
class PylidarVector
{
public:
    PylidarVector()
    {
        m_pData = NULL;
        m_nElems = 0;
        m_nTotalSize = 0;
        m_nGrowBy = 0;
        m_bOwned = false;
    }
    PylidarVector(npy_intp nStartSize, npy_intp nGrowBy)
    {
        m_pData = (T*)malloc(nStartSize * sizeof(T));
        m_nElems = 0;
        m_nTotalSize = nStartSize;
        m_nGrowBy = nGrowBy;
        m_bOwned = true;
    }
    ~PylidarVector()
    {
        reset();
    }

    void reset()
    {
        if( m_bOwned && ( m_pData != NULL) )
            free(m_pData);
        m_nElems = 0;
        m_nTotalSize = 0;
        m_nGrowBy = 0;
        m_bOwned = false;
        m_pData = NULL;
    }   

    npy_intp getNumElems()
    {
        return m_nElems;
    }

    void push(T *pNewElem)
    {
        if( !m_bOwned )
        {
            throw std::bad_alloc();
        }
        if( m_nElems == m_nTotalSize )
        {
            // realloc
            m_nTotalSize += m_nGrowBy;
            T *pNewData = (T*)realloc(m_pData, m_nTotalSize * sizeof(T));
            if( pNewData == NULL )
            {
                throw std::bad_alloc();
            }
            m_pData = pNewData;
        }
        memcpy(&m_pData[m_nElems], pNewElem, sizeof(T));
        m_nElems++;
    }

    T *getLastElement()
    {
        if(m_nElems == 0)
            return NULL;
        return &m_pData[m_nElems-1];
    }

    T *getFirstElement()
    {
        if(m_nElems == 0)
            return NULL;
        return &m_pData[0];
    }

    T *getElem(npy_intp n)
    {
        //if( n >= m_nElems )
        //    return NULL;
        return &m_pData[n];
    }

    void removeFront(npy_intp nRemove)
    {
        if( nRemove >= m_nElems )
        {
            // total removal
            m_nElems = 0;
        }
        else
        {
            // partial - first shuffle down
            npy_intp nRemain = m_nElems - nRemove;
            memmove(&m_pData[0], &m_pData[nRemove], nRemain * sizeof(T));
            m_nElems = nRemain;
        }
    }

    PyObject *getNumpyArray(SpylidarFieldDefn *pDefn)
    {
        // TODO: resize array down to nElems?
        m_bOwned = false;
        PyObject *p = pylidar_structArrayToNumpy(m_pData, m_nElems, pDefn);
        return p;
    }

    // split the upper part of this array into another
    // object. Takes elements from nUpper upwards into the 
    // new object.
    PylidarVector<T> *splitUpper(npy_intp nUpper)
    {
        if( !m_bOwned )
        {
            throw std::bad_alloc();
        }
        // split from nUpper to the end out as a new
        // PylidarVector
        npy_intp nNewSize = m_nElems - nUpper;
        PylidarVector<T> *splitted = new PylidarVector<T>(nNewSize, m_nGrowBy);
        memcpy(splitted->m_pData, &m_pData[nUpper], nNewSize * sizeof(T));
        splitted->m_nElems = nNewSize;

        // resize this one down
        m_nElems = nUpper;
        m_nTotalSize = nUpper;
        T *pNewData = (T*)realloc(m_pData, m_nTotalSize * sizeof(T));
        if( pNewData == NULL )
        {
            throw std::bad_alloc();
        }
        m_pData = pNewData;
        return splitted;
    }

    // split the lower part of this array into another
    // object. Takes elements from 0 up to (but not including) nUpper into the 
    // new object.
    PylidarVector<T> *splitLower(npy_intp nUpper)
    {
        if( !m_bOwned )
        {
            throw std::bad_alloc();
        }
        // split from 0 to nUpper as a new
        // PylidarVector
        npy_intp nNewSize = std::min(nUpper, m_nElems);
        PylidarVector<T> *splitted = new PylidarVector<T>(nNewSize, m_nGrowBy);
        memcpy(splitted->m_pData, &m_pData[0], nNewSize * sizeof(T));
        splitted->m_nElems = nNewSize;

        // resize this one down
        removeFront(nNewSize);

        return splitted;
    }

    void appendArray(PylidarVector<T> *other)
    {
        npy_intp nNewElems = m_nElems + other->m_nElems;
        npy_intp nNewTotalSize = m_nTotalSize;
        while( nNewTotalSize < nNewElems )
            nNewTotalSize += m_nGrowBy;

        if( nNewTotalSize > m_nTotalSize )
        {
            m_nTotalSize = nNewTotalSize;
            T *pNewData = (T*)realloc(m_pData, m_nTotalSize * sizeof(T));
            if( pNewData == NULL )
            {
                throw std::bad_alloc();
            }
            m_pData = pNewData;
        }

        memcpy(&m_pData[m_nElems], other->m_pData, other->m_nElems * sizeof(T));

        m_nElems = nNewElems;
    }

private:
    T *m_pData;
    bool m_bOwned;
    npy_intp m_nElems;
    npy_intp m_nTotalSize;
    npy_intp m_nGrowBy;
};

#endif

#endif /*PYLIDAR_H*/

