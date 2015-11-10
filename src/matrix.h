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

#ifndef MATRIX_H
#define MATRIX_H

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

    int getNumRows() const { return m_nRows; }
    int getNumCols() const { return m_nCols; }
    
private:
    int m_nRows;
    int m_nCols;
    T *m_pData;
};

} // namespace pylidar

#endif //MATRIX_H
