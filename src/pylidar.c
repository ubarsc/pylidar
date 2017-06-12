/*
 * pylidar.c
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

#include "pylidar.h"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* call first - preferably in the init() of your module
 this sets up the connection to numpy */
#if PY_MAJOR_VERSION >= 3
PyObject *pylidar_init()
#else
void pylidar_init()
#endif
{
    import_array();
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}

/* print error - used internally */
void pylidar_error(char *errstr, ...)
{
va_list args;

  va_start(args, errstr);
  vfprintf(stderr, errstr, args);
  va_end(args);
  fprintf(stderr, "\n");
}

/* Helper function to get information about a named field within an array
 pass null for params you not interested in */
int pylidar_getFieldDescr(PyArrayObject *pArray, const char *pszName, int *pnOffset, char *pcKind, int *pnSize, int *pnLength)
{
PyObject *pKey, *pValue;
Py_ssize_t pos = 0;
int bFound = 0;
#if PY_MAJOR_VERSION >= 3
PyObject *bytesKey;
#endif
char *pszElementName;
PyObject *pOffset;
PyArray_Descr *pSubDescr;
PyArray_Descr *pDescr;

    if( ! PyArray_Check(pArray) )
    {
        pylidar_error("Must pass array type");
        return 0;
    }

    pDescr = PyArray_DESCR(pArray);
    if( pDescr == NULL )
    {
        pylidar_error("Cannot get array description");
        return 0;
    }

    if( ( pDescr->byteorder != '|' ) && ( pDescr->byteorder != '=' ) )
    {
        pylidar_error("Cannot handle exotic byte order yet");
        return 0;
    }

    if( ( pDescr->fields == NULL ) || !PyDict_Check(pDescr->fields) )
    {
        pylidar_error("Cannot obtain the fields");
        return 0;
    }

    /* go through each of the fields looking for the right name
     see http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
     "this data-type-descriptor has fields described by a Python dictionary whose keys 
     are names (and also titles if given) and whose values are tuples that describe the fields."
     "A field is described by a tuple composed of another data- type-descriptor and a byte offset" */
    while( PyDict_Next(pDescr->fields, &pos, &pKey, &pValue) )
    {
#if PY_MAJOR_VERSION >= 3
        bytesKey = PyUnicode_AsEncodedString(pKey, NULL, NULL);
        pszElementName = PyBytes_AsString(bytesKey);
#else
        pszElementName = PyString_AsString(pKey);
#endif
        if( strcmp( pszElementName, pszName ) == 0 )
        {
            /* matches */
            bFound = 1;
            /* byte offset */
            pOffset = PyTuple_GetItem(pValue, 1);
            /* description */
            pSubDescr = (PyArray_Descr*)PyTuple_GetItem(pValue, 0);
            if( pSubDescr != NULL )
            {
                if( ( pSubDescr->kind == 'V' ) && ( pSubDescr->subarray != NULL ) )
                {
                    /* is a sub array */
                    if( pnOffset != NULL )
                    {
#if PY_MAJOR_VERSION >= 3
                        *pnOffset = PyLong_AsLong(pOffset);
#else
                        *pnOffset = PyInt_AsLong(pOffset);
#endif
                    }
                    if( pcKind != NULL )
                        *pcKind = pSubDescr->subarray->base->kind;
                    if( pnSize != NULL )
                        *pnSize = pSubDescr->subarray->base->elsize;
                    if( PyTuple_Size(pSubDescr->subarray->shape) != 1 )
                    {
                        pylidar_error("Can only handle 1-d sub arrays");
                        return 0;
                    }
                    if( pnLength != NULL )
                    {
#if PY_MAJOR_VERSION >= 3
                        *pnLength = PyLong_AsLong( PyTuple_GetItem(pSubDescr->subarray->shape, 0) );
#else
                        *pnLength = PyInt_AsLong( PyTuple_GetItem(pSubDescr->subarray->shape, 0) );
#endif
                    }
                }
                else
                {
                    /* is a single item */
                    if( pnOffset != NULL )
                    {
#if PY_MAJOR_VERSION >= 3
                        *pnOffset = PyLong_AsLong(pOffset);
#else
                        *pnOffset = PyInt_AsLong(pOffset);
#endif
                    }
                    if( pcKind != NULL )
                        *pcKind = pSubDescr->kind;
                    if( pnSize != NULL )
                        *pnSize = pSubDescr->elsize;
                    if( pnLength != NULL )
                        *pnLength = 1;
                }
            }
#if PY_MAJOR_VERSION >= 3
            Py_DECREF(bytesKey);
#endif
            break;
        }
#if PY_MAJOR_VERSION >= 3
        Py_DECREF(bytesKey);
#endif
    }
    if( !bFound )
    {
        pylidar_error("Couldn't find field %s", pszName);
        return 0;
    }
    return 1;
}

/* NOTE: Must to memcpy so works on RISC
 Use a macro to help cut down on cut and paste errors */
#define DO_INT64_LOOP(tempVar) for( nIdx = 0; nIdx < nArraySize; nIdx++ ) \
        { \
            pRow = PyArray_GETPTR1( pArray, nIdx ); \
            memcpy(&tempVar, (char*)pRow + nOffset, sizeof(tempVar)); \
            pOut[nIdx] = (npy_int64)tempVar; \
        } 


npy_int64 *pylidar_getFieldAsInt64(PyArrayObject *pArray, const char *pszName)
{
int nOffset, nSize, nLength;
char cKind;
npy_int64 *pOut;
npy_intp nArraySize, nIdx;
void *pRow;
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

    if( !pylidar_getFieldDescr(pArray, pszName, &nOffset, &cKind, &nSize, &nLength) )
    {
        return NULL;
    }

    if( nLength != 1 )
    {
        pylidar_error("Sub arrays not supported");
        return NULL;
    }

    if( PyArray_NDIM(pArray) != 1 )
    {
        pylidar_error("only 1 dimensional structured arrays supported");
        return NULL;
    }

    nArraySize = PyArray_DIM(pArray, 0);
    pOut = (npy_int64*)malloc(nArraySize * sizeof(npy_int64));
    if( pOut == NULL )
    {
        pylidar_error("memory alloc failed");
        return NULL;
    }

    /* decide on input type first so we can make loops fast */

    if( ( cKind == 'b' ) && ( nSize == 1 ) )
    {
        DO_INT64_LOOP(nBoolVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 1 ) )
    {
        DO_INT64_LOOP(nByteVal);
    }
    else if ( ( cKind == 'S' ) && ( nSize == 1 ) )
    {
        DO_INT64_LOOP(nCharVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 1 ) )
    {
        DO_INT64_LOOP(nUByteVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 2 ) )
    {
        DO_INT64_LOOP(nShortVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 2 ) )
    {
        DO_INT64_LOOP(nUShortVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 4 ) )
    {
        DO_INT64_LOOP(nIntVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 4 ) )
    {
        DO_INT64_LOOP(nUIntVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 8 ) )
    {
        DO_INT64_LOOP(nLongVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 8 ) )
    {
        DO_INT64_LOOP(nULongVal);
    }
    else if ( ( cKind == 'f' ) && ( nSize == 4 ) )
    {
        DO_INT64_LOOP(fFloatVal);
    }
    else if ( ( cKind == 'f' ) && ( nSize == 8 ) )
    {
        DO_INT64_LOOP(fDoubleVal);
    }

    return pOut;
}

/* Use a macro to help cut down on cut and paste errors */
#define DO_FLOAT64_LOOP(tempVar) for( nIdx = 0; nIdx < nArraySize; nIdx++ ) \
        { \
            pRow = PyArray_GETPTR1( pArray, nIdx ); \
            memcpy(&tempVar, (char*)pRow + nOffset, sizeof(tempVar)); \
            pOut[nIdx] = (double)tempVar; \
        } 

double *pylidar_getFieldAsFloat64(PyArrayObject *pArray, const char *pszName)
{
int nOffset, nSize, nLength;
char cKind;
double *pOut;
npy_intp nArraySize, nIdx;
void *pRow;
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

    if( !pylidar_getFieldDescr(pArray, pszName, &nOffset, &cKind, &nSize, &nLength) )
    {
        return NULL;
    }

    if( nLength != 1 )
    {
        pylidar_error("Sub arrays not supported");
        return NULL;
    }

    if( PyArray_NDIM(pArray) != 1 )
    {
        pylidar_error("only 1 dimensional structured arrays supported");
        return NULL;
    }

    nArraySize = PyArray_DIM(pArray, 0);
    pOut = (double*)malloc(nArraySize * sizeof(double));
    if( pOut == NULL )
    {
        pylidar_error("memory alloc failed");
        return NULL;
    }

    /* decide on input type first so we can make loops fast */

    if( ( cKind == 'b' ) && ( nSize == 1 ) )
    {
        DO_FLOAT64_LOOP(nBoolVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 1 ) )
    {
        DO_FLOAT64_LOOP(nByteVal);
    }
    else if ( ( cKind == 'S' ) && ( nSize == 1 ) )
    {
        DO_FLOAT64_LOOP(nCharVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 1 ) )
    {
        DO_FLOAT64_LOOP(nUByteVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 2 ) )
    {
        DO_FLOAT64_LOOP(nShortVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 2 ) )
    {
        DO_FLOAT64_LOOP(nUShortVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 4 ) )
    {
        DO_FLOAT64_LOOP(nIntVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 4 ) )
    {
        DO_FLOAT64_LOOP(nUIntVal);
    }
    else if ( ( cKind == 'i' ) && ( nSize == 8 ) )
    {
        DO_FLOAT64_LOOP(nLongVal);
    }
    else if ( ( cKind == 'u' ) && ( nSize == 8 ) )
    {
        DO_FLOAT64_LOOP(nULongVal);
    }
    else if ( ( cKind == 'f' ) && ( nSize == 4 ) )
    {
        DO_FLOAT64_LOOP(fFloatVal);
    }
    else if ( ( cKind == 'f' ) && ( nSize == 8 ) )
    {
        DO_FLOAT64_LOOP(fDoubleVal);
    }

    return pOut;
}

PyObject *pylidar_stringArrayToTuple(const char *data[])
{
    Py_ssize_t n;
    PyObject *pTuple;
    PyObject *pStr;
    const char *psz;

    /* how many do we have? */
    for( n = 0; data[n] != NULL; n++ )
    {
        /* do nothing */
    }

    /* now do it for real */
    pTuple = PyTuple_New(n);
    for( n = 0; data[n] != NULL; n++ )
    {
        psz = data[n];
#if PY_MAJOR_VERSION >= 3
        pStr = PyUnicode_FromString(psz);
#else
        pStr = PyString_FromString(psz);
#endif
        PyTuple_SetItem(pTuple, n, pStr);
    }

    return pTuple;
}


/* Wrap an existing C array of structures and return as a numpy array */
/* Python will free data when finished */
PyArrayObject *pylidar_structArrayToNumpy(void *pStructArray, npy_intp nElems, SpylidarFieldDefn *pDefn)
{
PyObject *pNameList, *pFormatList, *pOffsetList;
PyObject *pNameString, *pFormatString, *pOffsetInt, *pItemSizeObj;
PyObject *pDtypeDict, *pNamesKey, *pFormatsKey, *pOffsetsKey, *pItemSizeKey;
PyArray_Descr *pDescr;
PyArrayObject *pOut;
int nStructTotalSize = 0, i;
char *pszName;

    /* create a dictionary with all the info about the fields as 
     in http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html*/
    pNameList = PyList_New(0);
    pFormatList = PyList_New(0);
    pOffsetList = PyList_New(0);
    while( pDefn->pszName != NULL )
    {
        if( !pDefn->bIgnore )
        {
            /* Convert to upper case */
            pszName = strdup(pDefn->pszName);
            for( i = 0; pszName[i] != '\0'; i++ )
            {
                pszName[i] = toupper(pszName[i]);
            }
#if PY_MAJOR_VERSION >= 3
            pNameString = PyUnicode_FromString(pszName);
#else
            pNameString = PyString_FromString(pszName);
#endif
            PyList_Append(pNameList, pNameString);
            free(pszName);
            Py_DECREF(pNameString);

#if PY_MAJOR_VERSION >= 3
            pFormatString = PyUnicode_FromFormat("%c%d", (int)pDefn->cKind, pDefn->nSize);
#else
            pFormatString = PyString_FromFormat("%c%d", (int)pDefn->cKind, pDefn->nSize);
#endif
            PyList_Append(pFormatList, pFormatString);
            Py_DECREF(pFormatString);

#if PY_MAJOR_VERSION >= 3
            pOffsetInt = PyLong_FromLong(pDefn->nOffset);
#else
            pOffsetInt = PyInt_FromLong(pDefn->nOffset);
#endif
            PyList_Append(pOffsetList, pOffsetInt);
            Py_DECREF(pOffsetInt);

            nStructTotalSize = pDefn->nStructTotalSize;
        }

        pDefn++;
    }

    if( PyList_Size(pNameList) == 0)
    {
        Py_DECREF(pNameList);
        Py_DECREF(pFormatList);
        Py_DECREF(pOffsetList);
        pylidar_error("All fields were ignored");
        return NULL;
    }
            
    pDtypeDict = PyDict_New();
#if PY_MAJOR_VERSION >= 3
    pNamesKey = PyUnicode_FromString("names");
    pFormatsKey = PyUnicode_FromString("formats");
    pOffsetsKey = PyUnicode_FromString("offsets");
    pItemSizeKey = PyUnicode_FromString("itemsize");
#else
    pNamesKey = PyString_FromString("names");
    pFormatsKey = PyString_FromString("formats");
    pOffsetsKey = PyString_FromString("offsets");
    pItemSizeKey = PyString_FromString("itemsize");
#endif            
    PyDict_SetItem(pDtypeDict, pNamesKey, pNameList);
    Py_DECREF(pNameList);
    PyDict_SetItem(pDtypeDict, pFormatsKey, pFormatList);
    Py_DECREF(pFormatList);
    PyDict_SetItem(pDtypeDict, pOffsetsKey, pOffsetList);
    Py_DECREF(pOffsetList);
#if PY_MAJOR_VERSION >= 3
    pItemSizeObj = PyLong_FromLong(nStructTotalSize);
#else
    pItemSizeObj = PyInt_FromLong(nStructTotalSize);
#endif
    PyDict_SetItem(pDtypeDict, pItemSizeKey, pItemSizeObj);
    Py_DECREF(pItemSizeObj);

    /* Convert dictionary to array description */
    if( !PyArray_DescrConverter(pDtypeDict, &pDescr) )
    {
        pylidar_error("Unable to convert array description");
        return NULL;
    }

    Py_DECREF(pDtypeDict);

    /* Create new array wrapping the existing C one */
    pOut = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, pDescr, 1, &nElems, NULL, pStructArray,
                    NPY_ARRAY_CARRAY, NULL );
    if( pOut == NULL )
    {
        pylidar_error("Unable to create array");
        return NULL;
    }

    /* Need to set this separately since PyArray_NewFromDescr always */
    /* has it unset, even if you pass it as the flags parameter */
    PyArray_ENABLEFLAGS(pOut, NPY_ARRAY_OWNDATA);

    return pOut;
}

PyArray_Descr *pylidar_getDtypeForField(SpylidarFieldDefn *pDefn, const char *pszFieldname)
{
PyArray_Descr *pDescr = NULL;
PyObject *pString;
int i;
char *pszName;

    while( pDefn->pszName != NULL )
    {
        /* Convert to upper case */
        pszName = strdup(pDefn->pszName);
        for( i = 0; pszName[i] != '\0'; i++ )
        {
            pszName[i] = toupper(pszName[i]);
        }

        if( strcmp(pszName, pszFieldname) == 0 )
        {
            /* Now build dtype string - easier than having a switch on all the combinations */
#if PY_MAJOR_VERSION >= 3
            pString = PyUnicode_FromFormat("%c%d", pDefn->cKind, pDefn->nSize);
#else
            pString = PyString_FromFormat("%c%d", pDefn->cKind, pDefn->nSize);
#endif
            /* assume success */
            PyArray_DescrConverter(pString, &pDescr);
            Py_DECREF(pString);
            break;
        }

        free(pszName);

        pDefn++;
    }

    return pDescr;
}

/* Set the ignore state of a named field in pDefn */
int pylidar_setIgnore(SpylidarFieldDefn *pDefn, const char *pszFieldname, char bIgnore)
{
    int bFound = 0;
    while( pDefn->pszName != NULL )
    {
        /* should this be case insensitive? */
        if( strcmp(pDefn->pszName, pszFieldname) == 0 )
        {
            pDefn->bIgnore = bIgnore;
            bFound = 1;
            break;
        }

        pDefn++;
    }

    if( !bFound )
        fprintf(stderr, "Field %s not found in pylidar_setIgnore\n", pszFieldname);

    return bFound;
}

