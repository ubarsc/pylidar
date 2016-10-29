/*
 * pylvector.h
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

#ifndef PYLVECTOR_H
#define PYLVECTOR_H

#include "pylidar.h"
#include <new>
#include <algorithm>

namespace pylidar
{

// A class for managing a dynamic number of elements
// and creating a numpy array of them if needed.
template <class T>
class CVector
{
public:
    CVector()
    {
        m_pData = NULL;
        m_nElems = 0;
        m_nTotalSize = 0;
        m_nGrowBy = 0;
        m_bOwned = false;
        m_nElemSize = 0;
    }
    // Note: you can set nElemSize if you don't know size of type at compile time
    CVector(npy_intp nStartSize, npy_intp nGrowBy, npy_intp nElemSize=sizeof(T))
    {
        m_pData = (T*)PyDataMem_NEW(nStartSize * nElemSize);
        m_nElems = 0;
        m_nTotalSize = nStartSize;
        m_nGrowBy = nGrowBy;
        m_bOwned = true;
        m_nElemSize = nElemSize;
    }
    // takes copy
    CVector(T *pData, npy_intp nSize, npy_intp nGrowBy=1)
    {
        m_pData = (T*)PyDataMem_NEW(nSize);
        memcpy(m_pData, pData, nSize);
        m_nElems = nSize / sizeof(T);
        m_nTotalSize = m_nElems;
        m_nGrowBy = nGrowBy;
        m_bOwned = true;
        m_nElemSize = sizeof(T);
    }
    ~CVector()
    {
        reset();
    }

    void reset()
    {
        if( m_bOwned && ( m_pData != NULL) )
            PyDataMem_FREE((char*)m_pData);
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
            T *pNewData = (T*)PyDataMem_RENEW(m_pData, m_nTotalSize * m_nElemSize);
            if( pNewData == NULL )
            {
                throw std::bad_alloc();
            }
            m_pData = pNewData;
        }
        memcpy(getElem(m_nElems), pNewElem, m_nElemSize);
        m_nElems++;
    }

    T *getLastElement()
    {
        if(m_nElems == 0)
            return NULL;
        return getElem(m_nElems-1);
    }

    T *getFirstElement()
    {
        if(m_nElems == 0)
            return NULL;
        return getElem(0);
    }

    T *getElem(npy_intp n)
    {
        //if( n >= m_nElems )
        //    return NULL;
        // NB: sizeof(T) could be != m_nElemSize
        return (T*)((char*)m_pData + (m_nElemSize * n));
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
            memcpy(getElem(0), getElem(nRemove), nRemain * m_nElemSize);
            m_nElems = nRemain;
        }
    }

    // for structured arrays
    PyArrayObject *getNumpyArray(SpylidarFieldDefn *pDefn)
    {
        // TODO: resize array down to nElems?
        m_bOwned = false;
        if( m_nElems > 0 )
        {
            return pylidar_structArrayToNumpy(m_pData, m_nElems, pDefn);
        }
        else
        {
            // free mem, otherwise we get a memory leak as numpy 
            // doesn't seem to free a empty array
            PyDataMem_FREE((char*)m_pData);
            return pylidar_structArrayToNumpy(NULL, 0, pDefn);
        }
    }

    // for non structured arrays
    PyArrayObject *getNumpyArray(int typenum)
    {
        m_bOwned = false;
        npy_intp dims = m_nElems;
        PyArrayObject *p = NULL;
        if( m_nElems > 0 )
        {
            p = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, typenum, (void*)m_pData);
            PyArray_ENABLEFLAGS(p, NPY_ARRAY_OWNDATA);
        }
        else
        {
            // free mem, otherwise we get a memory leak as numpy 
            // doesn't seem to free a empty array
            PyDataMem_FREE((char*)m_pData);
            p = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, typenum, NULL);
        }
        return p;
    }

    // split the upper part of this array into another
    // object. Takes elements from nUpper upwards into the 
    // new object.
    CVector<T> *splitUpper(npy_intp nUpper)
    {
        if( !m_bOwned )
        {
            throw std::bad_alloc();
        }
        // split from nUpper to the end out as a new
        // CVector
        npy_intp nNewSize = m_nElems - nUpper;
        CVector<T> *splitted = new CVector<T>(nNewSize, m_nGrowBy);
        memcpy(splitted->m_pData, getElem(nUpper), nNewSize * m_nElemSize);
        splitted->m_nElems = nNewSize;

        // resize this one down
        m_nElems = nUpper;
        m_nTotalSize = nUpper;
        T *pNewData = (T*)PyDataMem_RENEW(m_pData, m_nTotalSize * m_nElemSize);
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
    CVector<T> *splitLower(npy_intp nUpper)
    {
        if( !m_bOwned )
        {
            throw std::bad_alloc();
        }
        // split from 0 to nUpper as a new
        // CVector
        npy_intp nNewSize = std::min(nUpper, m_nElems);
        CVector<T> *splitted = new CVector<T>(nNewSize, m_nGrowBy);
        memcpy(splitted->m_pData, getElem(0), nNewSize * m_nElemSize);
        splitted->m_nElems = nNewSize;

        // resize this one down
        removeFront(nNewSize);

        return splitted;
    }

    void appendArray(CVector<T> *other)
    {
        npy_intp nNewElems = m_nElems + other->m_nElems;
        npy_intp nNewTotalSize = m_nTotalSize;
        while( nNewTotalSize < nNewElems )
            nNewTotalSize += m_nGrowBy;

        if( nNewTotalSize > m_nTotalSize )
        {
            m_nTotalSize = nNewTotalSize;
            T *pNewData = (T*)PyDataMem_RENEW(m_pData, m_nTotalSize * m_nElemSize);
            if( pNewData == NULL )
            {
                throw std::bad_alloc();
            }
            m_pData = pNewData;
        }

        memcpy(getElem(m_nElems), other->m_pData, other->m_nElems * m_nElemSize);

        m_nElems = nNewElems;
    }

private:
    T *m_pData;
    bool m_bOwned;
    npy_intp m_nElems;
    npy_intp m_nTotalSize;
    npy_intp m_nGrowBy;
    npy_intp m_nElemSize; // normall sizeof(T) but can be overidden
};

} //namespace pylidar

#endif //PYLVECTOR_H

