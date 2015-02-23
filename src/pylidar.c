
#include "pylidar.h"

#include "numpy/arrayobject.h"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* call first - preferably in the init() of your module
 this sets up the connection to numpy */
void pylidar_init()
{
    import_array();
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
int pylidar_getFieldDescr(PyObject *pArray, const char *pszName, int *pnOffset, char *pcKind, int *pnSize, int *pnLength)
{
PyObject *pKey, *pValue;
Py_ssize_t pos = 0;
int bFound = 0;
PyObject *bytesKey;
char *pszElementName;
PyObject *pOffset;
PyArray_Descr *pSubDescr;

    if( ! PyArray_Check(pArray) )
    {
        pylidar_error("Must pass array type");
        return 0;
    }

    PyArray_Descr *pDescr = PyArray_DESCR(pArray);
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
                    }
#endif
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
                    }
#endif
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
            pOut[nIdx] = (int64_t)tempVar; \
        } 


int64_t *pylidar_getFieldAsInt64(PyObject *pArray, const char *pszName)
{
int nOffset, nSize, nLength;
char cKind;
int64_t *pOut;
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
    pOut = (int64_t*)malloc(nArraySize * sizeof(int64_t));
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

double *pylidar_getFieldAsFloat64(PyObject *pArray, const char *pszName)
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

/* Wrap an existing C array of structures and return as a numpy array */
/* Python will free data when finished */
PyObject *pylidar_structArrayToNumpy(void *pStructArray, npy_intp nElems, SpylidarFieldDefn *pDefn)
{
PyObject *pNameList, *pFormatList, *pOffsetList;
PyObject *pNameString, *pFormatString, *pOffsetInt;
PyObject *pDtypeDict, *pNamesKey, *pFormatsKey, *pOffsetsKey, *pItemSizeKey;
PyArray_Descr *pDescr;
PyObject *pOut;
int nStructTotalSize;

    /* create a dictionary with all the info about the fields as 
     in http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html*/
    pNameList = PyList_New(0);
    pFormatList = PyList_New(0);
    pOffsetList = PyList_New(0);
    while( pDefn->pszName != NULL )
    {
#if PY_MAJOR_VERSION >= 3
        pNameString = PyUnicode_FromString(pDefn->pszName);
#else
        pNameString = PyString_FromString(pDefn->pszName);
#endif
        PyList_Append(pNameList, pNameString);

#if PY_MAJOR_VERSION >= 3
        pFormatString = PyUnicode_FromFormat("%c%d", (int)pDefn->cKind, pDefn->nSize);
#else
        pFormatString = PyString_FromFormat("%c%d", (int)pDefn->cKind, pDefn->nSize);
#endif
        PyList_Append(pFormatList, pFormatString);

        pOffsetInt = PyInt_FromLong(pDefn->nOffset);
        PyList_Append(pOffsetList, pOffsetInt);

        nStructTotalSize = pDefn->nStructTotalSize;

        pDefn++;
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
    PyDict_SetItem(pDtypeDict, pFormatsKey, pFormatList);
    PyDict_SetItem(pDtypeDict, pOffsetsKey, pOffsetList);
    PyDict_SetItem(pDtypeDict, pItemSizeKey, PyInt_FromLong(nStructTotalSize) );

    /* Convert dictionary to array description */
    if( !PyArray_DescrConverter(pDtypeDict, &pDescr) )
    {
        pylidar_error("Unable to convert array description");
        return NULL;
    }

    /* Create new array wrapping the existing C one */
    pOut = PyArray_NewFromDescr(&PyArray_Type, pDescr, 1, &nElems, NULL, pStructArray,
                    NPY_ARRAY_CARRAY, NULL );
    if( pOut == NULL )
    {
        pylidar_error("Unable to create array");
        return NULL;
    }
    return pOut;
}
