/*
 * matrix.h
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

#ifndef PYLMATRIX_H
#define PYLMATRIX_H

// This class implements simple matrix addition and 
// multiplication.
// Linking in BLAS etc seemed overkill for the tiny (4x4)
// matrix sizes used in the C++ driver parts of PyLidar
// hence this small (and probably inefficient) code.

#include <stdlib.h>
#include <string.h>
#include <stdexcept>

namespace pylidar
{

template <class T>
class CMatrix
{
public:
    CMatrix(const CMatrix<T> &other)
    {
        m_nRows = other.m_nRows;
        m_nCols = other.m_nCols;
        m_pData = (T*)malloc(sizeof(T) * m_nRows * m_nCols);
        memcpy(m_pData, other.m_pData, sizeof(T) * m_nRows * m_nCols);
    }
    CMatrix(int nRows, int nCols)
    {
        m_nRows = nRows;
        m_nCols = nCols;
        m_pData = (T*)malloc(sizeof(T) * nRows * nCols);
    }
    // assumed type matches T - no checking happens
    // apart from a rough check on size
    CMatrix(PyArrayObject *pNumpyArray)
    {
        if( PyArray_NDIM(pNumpyArray) != 2)
        {
            throw std::length_error("numpy array must be 2 dimensional");
        }
        if( PyArray_ITEMSIZE(pNumpyArray) != sizeof(T) )
        {
            throw std::invalid_argument("numpy array elements not same size as matrix type");
        }
        
        m_nRows = PyArray_DIM(pNumpyArray, 0);
        m_nCols = PyArray_DIM(pNumpyArray, 1);
        m_pData = (T*)malloc(sizeof(T) * m_nRows * m_nCols);
        T *p;
        for(int r = 0; r < m_nRows; ++r)
        {
            for(int c = 0; c < m_nCols; ++c)
            {
                p = (T*)PyArray_GETPTR2(pNumpyArray, r, c);
                set(r, c, *p);
            }
        }
    }
    ~CMatrix()
    {
        free(m_pData);
    }

    T get(int row, int col) const
    {
        T *p = (m_pData + (row * m_nCols)) + col;
        return *p;
    }
    void set(int row, int col, T v)
    {
        T *p = (m_pData + (row * m_nCols)) + col;
        *p = v;
    }

    CMatrix<T> add(CMatrix<T> &other) const
    {
        if( (m_nRows != other.m_nRows) || (m_nCols != other.m_nCols) )
        {
            throw std::length_error("matrices must be same size");
        }

        CMatrix<T> result(m_nRows, m_nCols);
        T v;
        for(int r = 0; r < m_nRows; ++r)
        {
            for(int c = 0; c < m_nCols; ++c)
            {
                v = get(r, c) + other.get(r, c);
                result.set(r, c, v);
            }
        }

        return result;
    }

    CMatrix multiply(CMatrix &other) const
    {
        if( m_nCols != other.m_nRows )
        {
            throw std::length_error("number of cols in a must equal number of rows in b");
        }

        CMatrix<T> result(m_nRows, other.m_nCols);
        T v;
        for(int r = 0; r < m_nRows; ++r)
        {
            for(int c = 0; c < other.m_nCols; ++c)
            {
                v = 0;
                for(int k = 0; k < m_nCols; ++k)
                {
                    v += get(r, k) * other.get(k, c);
                }
                result.set(r, c, v);
            }
        }

        return result;
    }

    // return a new numpy array with the contents
    // of this object. typenum should correspond to T.
    PyArrayObject *getAsNumpyArray(int typenum=NPY_FLOAT) const
    {
        npy_intp dims[2];
        dims[0] = m_nRows;
        dims[1] = m_nCols;
        PyArrayObject *pNumpyArray = (PyArrayObject*)PyArray_SimpleNew(2, dims, typenum);
        T *p;
        for(int r = 0; r < m_nRows; ++r)
        {
            for(int c = 0; c < m_nCols; ++c)
            {
                p = (T*)PyArray_GETPTR2(pNumpyArray, r, c);
                *p = get(r, c);
            }
        }
        return pNumpyArray;
    }

    int getNumRows() const { return m_nRows; }
    int getNumCols() const { return m_nCols; }
    
private:
    int m_nRows;
    int m_nCols;
    T *m_pData;
};

} // namespace pylidar

#endif //PYLMATRIX_H
