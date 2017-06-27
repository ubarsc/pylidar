/*
 * ascii.cpp
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <new>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "pylvector.h"

#ifdef HAVE_ZLIB
    #include "zlib.h"
#endif

#ifdef _MSC_VER
    // MSVC not happy about fopen, just shut it up for now
    #define _CRT_SECURE_NO_WARNINGS
#endif

// for CVector
static const int nGrowBy = 10000;
static const int nInitSize = 256*256;

static const int nMaxLineSize = 8192;

/* An exception object for this module */
/* created in the init function */
struct ASCIIState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct ASCIIState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
static struct ASCIIState _state;
#endif

// file compression types
#define ASCII_UNKNOWN 0
#define ASCII_UNCOMPRESSED 1
#define ASCII_GZIP 2

static PyObject *ascii_getFileType(PyObject *self, PyObject *args)
{
    const char *pszFileName;
    if( !PyArg_ParseTuple(args, "s:getFileType", &pszFileName))
        return NULL;

    int nLen = strlen(pszFileName);
    int i = nLen - 1;
    while( ( i >= 0) && (pszFileName[i] != '.' ) )
        i--;

    // no ext
    if( i < 0 )
        i = nLen;

    // take a copy of the extension part so we can make
    // it lower case
    char *pszLastExt = strdup(&pszFileName[i]);
    i = 0;
    while( pszLastExt[i] != '\0' )
    {
        pszLastExt[i] = tolower(pszLastExt[i]);
        i++;
    }

    int nType = ASCII_UNKNOWN;
    FILE *pFH = fopen(pszFileName, "rb");
    if( pFH == NULL )
    {
        PyErr_Format(GETSTATE(self)->error, "Unable to open file: %s", pszFileName);
        return NULL;
    }

    unsigned char aData[2];
    // read the first 2 bytes and confirm gzip header
    if( fread(aData, 1, sizeof(aData), pFH) != sizeof(aData) )
    {
        PyErr_Format(GETSTATE(self)->error, "Cannot read file: %s", pszFileName);
        fclose(pFH);
        return NULL;
    }
    fclose(pFH);

    // gzip?
    if( ( nLen >= 3 ) && (strcmp(pszLastExt, ".gz") == 0) )
    {
        // check for gzip header
        if( ( aData[0] == 0x1f ) && ( aData[1] == 0x8b ) )
        {
            nType = ASCII_GZIP;
        }
    }

    // not gzip. Try uncompressed
    if( (nType == ASCII_UNKNOWN ) && ( nLen >= 4 ) && ((strcmp(pszLastExt, ".dat") == 0 ) ||
            (strcmp(pszLastExt, ".csv") == 0 ) || (strcmp(pszLastExt, ".txt") == 0 )) )
    {
        // just check first char is a digit or a space or a comment
        // TODO: should we able to configure what is a comment like we do below in the reader?
        // not always able to - eg from file info we don't have the driver opyions
        if( (aData[0] == ' ') || isdigit(aData[0]) || (aData[0] == '#') )
        {
            nType = ASCII_UNCOMPRESSED;
        }
    }

    // insert other tests here

    free(pszLastExt);

    if( nType == ASCII_UNKNOWN )
    {
        PyErr_Format(GETSTATE(self)->error, "Don't know how to read %s", pszFileName);
        return NULL;
    }

    return PyLong_FromLong(nType);
}

// module methods
static PyMethodDef module_methods[] = {
    {"getFileType", (PyCFunction)ascii_getFileType, METH_VARARGS,
        "Determine the file type, raises exception if not understood"},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static int ascii_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int ascii_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_ascii",
        NULL,
        sizeof(struct ASCIIState),
        module_methods,
        NULL,
        ascii_traverse,
        ascii_clear,
        NULL
};
#endif

/* Python object wrapping a file reader */
typedef struct 
{
    PyObject_HEAD

    // one of these will be non-null
#ifdef HAVE_ZLIB
    gzFile  gz_file;
#endif
    FILE    *unc_file; // uncompressed

    Py_ssize_t nPulsesRead;
    bool bFinished;
    bool bTimeSequential;
    char cCommentChar;

    int nPulseFields;
    SpylidarFieldDefn *pPulseDefn;
    int *pPulseLineIdxs;

    int nPointFields;
    SpylidarFieldDefn *pPointDefn;
    int *pPointLineIdxs;

} PyASCIIReader;

void FreeDefn(SpylidarFieldDefn *pDefn, int nFields)
{
    for( int i = 0; i < nFields; i++ )
    {
        if( pDefn[i].pszName != NULL )
            free((void*)pDefn[i].pszName);
    }
    free(pDefn);
}

// pList is a list of (name, dtype, idx) tuples
// pnFields will be set to the count of fields (nPulseFields/nPointFields)
// ppnIdxs will be set to an array of idxs (pPulseLineIdxs/pPointLineIdxs)
// If bInsertPulseFields in true then NUMBER_OF_RETURNS will be added PTS_START_IDX
//      to the definition but not into pnFields or ppnIdxs
SpylidarFieldDefn *DTypeListToFieldDef(PyObject *pList, int *pnFields, int **ppnIdxs,
        bool bInsertPulseFields, PyObject *error)
{
    if( !PySequence_Check(pList) )
    {
        PyErr_SetString(error, "Parameter is not a sequence");
        return NULL;
    }

    Py_ssize_t nSize = PySequence_Size(pList);
    // create Defn
    Py_ssize_t nDefnSize = nSize;
    if( bInsertPulseFields)
        nDefnSize += 2;

    SpylidarFieldDefn *pDefn = NULL;
    if( nDefnSize > 0 )
    {
        // +1 for sentinel
        pDefn = (SpylidarFieldDefn*)calloc(nDefnSize+1, sizeof(SpylidarFieldDefn));
    }
    else
    {
        PyErr_SetString(error, "No definitions found");
        return NULL;
    }
    // idxs
    if( nSize > 0 )
        *ppnIdxs = (int*)calloc(nSize, sizeof(int));
    else
        *ppnIdxs = NULL;

    int nOffset = 0;
    for( Py_ssize_t i = 0; i < nSize; i++ )
    {
        PyObject *pElem = PySequence_GetItem(pList, i);
        if( !PySequence_Check(pElem) || ( PySequence_Size(pElem) != 3 ) )
        {
            PyErr_SetString(error, "Each element must be a 3 element sequence");
            Py_DECREF(pElem);
            FreeDefn(pDefn, nSize);
            free(*ppnIdxs);
            *ppnIdxs = NULL;
            return NULL;
        }

        PyObject *pName = PySequence_GetItem(pElem, 0);
#if PY_MAJOR_VERSION >= 3
        if( !PyUnicode_Check(pName) )
#else
        if( !PyString_Check(pName ) )
#endif
        {
            PyErr_SetString(error, "First element must be string");
            Py_DECREF(pElem);
            Py_DECREF(pName);
            FreeDefn(pDefn, nSize);
            free(*ppnIdxs);
            *ppnIdxs = NULL;
            return NULL;
        }

#if PY_MAJOR_VERSION >= 3
        PyObject *bytesKey = PyUnicode_AsEncodedString(pName, NULL, NULL);
        pDefn[i].pszName = strdup(PyBytes_AsString(bytesKey));
        Py_DECREF(bytesKey);
#else
        pDefn[i].pszName = strdup(PyString_AsString(pName));
#endif
        Py_DECREF(pName);

        PyObject *pNumpyDType = PySequence_GetItem(pElem, 1);
        // we can't actually do much with numpy.uint16 etc
        // which is what is passed in. Turn into numpy descr for more info
        PyArray_Descr *pDescr = NULL;
        if( !PyArray_DescrConverter(pNumpyDType, &pDescr) )
        {
            PyErr_SetString(error, "Couldn't convert 2nd element type to numpy Descr");
            Py_DECREF(pElem);
            Py_DECREF(pNumpyDType);
            FreeDefn(pDefn, nSize);
            free(*ppnIdxs);
            *ppnIdxs = NULL;
            return NULL;
        }

        pDefn[i].cKind = pDescr->kind;
        pDefn[i].nSize = pDescr->elsize;
        pDefn[i].nOffset = nOffset;
        nOffset += pDescr->elsize;
        // do nStructTotalSize last once we have all the sizes

        Py_DECREF(pDescr);
        Py_DECREF(pNumpyDType);

        // now for the idxs
        PyObject *pIdx = PySequence_GetItem(pElem, 2);
        PyObject *pIdxLong = PyNumber_Long(pIdx);
        if( !PyLong_Check(pIdxLong) )
        {
            PyErr_SetString(error, "3rd element must be int");
            Py_DECREF(pIdx);
            Py_DECREF(pIdxLong);
            Py_DECREF(pElem);
            FreeDefn(pDefn, nSize);
            free(*ppnIdxs);
            *ppnIdxs = NULL;
            return NULL;
        }

        (*ppnIdxs)[i] = PyLong_AsLong(pIdxLong);
        Py_DECREF(pIdx);
        Py_DECREF(pIdxLong);

        Py_DECREF(pElem);
    }

    if( bInsertPulseFields )
    {
        pDefn[nSize].pszName = strdup("NUMBER_OF_RETURNS");
        pDefn[nSize].cKind = 'u';
        pDefn[nSize].nSize = 1;
        pDefn[nSize].nOffset = nOffset;
        nOffset += pDefn[nSize].nSize;

        pDefn[nSize+1].pszName = strdup("PTS_START_IDX");
        pDefn[nSize+1].cKind = 'u';
        pDefn[nSize+1].nSize = 8;
        pDefn[nSize+1].nOffset = nOffset;
        nOffset += pDefn[nSize+1].nSize;
    }

    // now do nStructTotalSize
    for( Py_ssize_t i = 0; i < nDefnSize; i++ )
    {
        pDefn[i].nStructTotalSize = nOffset;
    }

    *pnFields = nSize;
    return pDefn;
}

/* init method - open file */
static int 
PyASCIIReader_init(PyASCIIReader *self, PyObject *args, PyObject *kwds)
{
    const char *pszFname = NULL, *pszCommentChar = NULL;
    int nType, nTimeSequential;
    PyObject *pPulseDTypeList, *pPointDTypeList;

    if( !PyArg_ParseTuple(args, "siOOis", &pszFname, &nType, &pPulseDTypeList,
                &pPointDTypeList, &nTimeSequential, &pszCommentChar ) )
    {
        return -1;
    }

    PyObject *error = GETSTATE_FC->error;

    // nType should come from getFileType()
#ifdef HAVE_ZLIB
    self->gz_file = NULL;
#endif
    self->unc_file = NULL;

    if( nType == ASCII_GZIP )
    {
#ifdef HAVE_ZLIB
        self->gz_file = gzopen(pszFname, "rb");
        if( self->gz_file == NULL )
        {
            PyErr_SetString(error, "Unable to open file");
            return -1;
        }
#else
        PyErr_SetString(error, "GZIP files need zlib library. ZLIB_ROOT environment variable should be set when building pylidar\n");
        return -1;
#endif
    }
    else if( nType == ASCII_UNCOMPRESSED )
    {
        self->unc_file = fopen(pszFname, "r");
        if( self->unc_file == NULL )
        {
            PyErr_SetString(error, "Unable to open file");
            return -1;
        }
    }
    else
    {
        PyErr_SetString(error, "type parameter not understood. Should be that returned from getFileType()");
        return -1;
    }

    self->nPulsesRead = 0;
    self->bFinished = false;
    if( nTimeSequential )
        self->bTimeSequential = true;
    else
        self->bTimeSequential = false;

    self->cCommentChar = '\0';
    // if empty turns this feature off
    if( strlen(pszCommentChar) > 0 )
        self->cCommentChar = pszCommentChar[0]; 

    self->pPulseLineIdxs = NULL;
    self->pPointLineIdxs = NULL;

    // create our definitions

    self->pPulseDefn = DTypeListToFieldDef(pPulseDTypeList, &self->nPulseFields, 
            &self->pPulseLineIdxs, true, error);
    if( self->pPulseDefn == NULL )
    {
        // error should be set
        return -1;
    }

    self->pPointDefn = DTypeListToFieldDef(pPointDTypeList, &self->nPointFields, 
            &self->pPointLineIdxs, false, error);
    if( self->pPointDefn == NULL )
    {
        // error should be set
        return -1;
    }

    return 0;
}

/* destructor - close and delete */
static void 
PyASCIIReader_dealloc(PyASCIIReader *self)
{
#ifdef HAVE_ZLIB
    if( self->gz_file != NULL )
    {
        gzclose(self->gz_file);
    }
#endif
    if( self->unc_file != NULL )
    {
        fclose(self->unc_file);
    }
    if( self->pPulseDefn != NULL )
    {
        FreeDefn( self->pPulseDefn, self->nPulseFields);
    }
    if( self->pPointDefn != NULL )
    {
        FreeDefn( self->pPointDefn, self->nPointFields);
    }
    if( self->pPulseLineIdxs != NULL )
    {
        free(self->pPulseLineIdxs);
    }
    if( self->pPointLineIdxs != NULL )
    {
        free(self->pPointLineIdxs);
    }
}

class CReadState
{
public:
    CReadState(int nFields, char cCommentChar)
    {
        m_bFirst = true;
        m_nFields = nFields;
        m_cCommentChar = cCommentChar;
        m_pszCurrentLine = (char*)malloc(nMaxLineSize * sizeof(char));
        if( m_pszCurrentLine == NULL )
        {
            throw std::bad_alloc();
        }
        m_pszLastLine = (char*)malloc(nMaxLineSize * sizeof(char));
        if( m_pszLastLine == NULL )
        {
            free(m_pszCurrentLine);
            throw std::bad_alloc();
        }

        m_pnCurrentColIdxs = (int*)malloc(nFields * sizeof(int));
        if( m_pnCurrentColIdxs == NULL )
        {
            free(m_pszCurrentLine);
            free(m_pszLastLine);
            throw std::bad_alloc();
        }
        m_pnLastColIdxs = (int*)malloc(nFields * sizeof(int));
        if( m_pnLastColIdxs == NULL )
        {
            free(m_pszCurrentLine);
            free(m_pszLastLine);
            free(m_pnCurrentColIdxs);
            throw std::bad_alloc();
        }
    }
    ~CReadState()
    {
        free(m_pszCurrentLine);
        free(m_pszLastLine);
        free(m_pnCurrentColIdxs);
        free(m_pnLastColIdxs);
    }

    bool getNewLine(PyASCIIReader *self, const char **pszErrorString)
    {
        // read into last line then clobber
        // try the various reader handles...
        bool bIsComment = true;
        int nStartIdx;
        while(bIsComment)
        {
#ifdef HAVE_ZLIB
            if( self->gz_file != NULL )
            {
                if( gzgets(self->gz_file, m_pszLastLine, nMaxLineSize) == NULL)
                {
                    return false;
                }
            }
#endif
            if( self->unc_file != NULL )
            {
                if( fgets(m_pszLastLine, nMaxLineSize, self->unc_file) == NULL)
                {
                    return false;
                }
            }

            // find first idx, some files have spaces etc at the start of the line
            nStartIdx = 0;
            while(isspace(m_pszLastLine[nStartIdx]) && (nStartIdx < nMaxLineSize))
                nStartIdx++;

            bIsComment = (m_pszLastLine[nStartIdx] == m_cCommentChar);
        }

        char *pszOldCurrent = m_pszCurrentLine;
        m_pszCurrentLine = m_pszLastLine;
        m_pszLastLine = pszOldCurrent;

        // now do the indices
        int *pnOldLast = m_pnLastColIdxs;
        m_pnLastColIdxs = m_pnCurrentColIdxs;
        m_pnCurrentColIdxs = pnOldLast;

        m_pnCurrentColIdxs[0] = nStartIdx;
        for( int i = 1; i < m_nFields; i++ )
        {
            // go through all the numbers
            while((isdigit(m_pszCurrentLine[nStartIdx]) || (m_pszCurrentLine[nStartIdx] == '.') || (m_pszCurrentLine[nStartIdx] == '-')) 
                        && (nStartIdx < nMaxLineSize))
                nStartIdx++;

            // then anything in between - setting to \0 while we are at it
            while(((m_pszCurrentLine[nStartIdx] == ' ') || (m_pszCurrentLine[nStartIdx] == ',') || (m_pszCurrentLine[nStartIdx] == '\t')) 
                        && (nStartIdx < nMaxLineSize))
            {
                m_pszCurrentLine[nStartIdx] = '\0';
                nStartIdx++;
            }

            if( (m_pszCurrentLine[nStartIdx] == '\n' ) || (m_pszCurrentLine[nStartIdx] == '\r'))
            {
                *pszErrorString = "Number of columns in file does not match expected";
                return false;
            }

            m_pnCurrentColIdxs[i] = nStartIdx;
        }

        // keep going so we can replace \n with \0
        while((m_pszCurrentLine[nStartIdx] != '\n') && (m_pszCurrentLine[nStartIdx] != '\r') && (nStartIdx < nMaxLineSize))
            nStartIdx++;

        if( (m_pszCurrentLine[nStartIdx] == '\n') || (m_pszCurrentLine[nStartIdx] == '\r'))
            m_pszCurrentLine[nStartIdx] = '\0';

        // print
        /*
        fprintf(stderr, "nfields = %d\n", m_nFields);
        fprintf(stderr, "Current:\n{");
        for( int i = 0; i < m_nFields; i++)
        {
            fprintf(stderr, "%d=%s - ", m_pnCurrentColIdxs[i], &m_pszCurrentLine[m_pnCurrentColIdxs[i]]);
        }
        if( !m_bFirst )
        {
            fprintf(stderr, "}\nLast:\n{");
            for( int i = 0; i < m_nFields; i++)
            {
                fprintf(stderr, "%d=%s - ", m_pnLastColIdxs[i], &m_pszLastLine[m_pnLastColIdxs[i]]);
            }
        }
        fprintf(stderr, "}\n");*/
        m_bFirst = false;
        return true;
    }

    bool isSamePulse(int *pPulseIdxs, int nPulseIdxs)
    {
        if( m_bFirst )
            return false;  // this ok?

        // check all the strings are the same
        for( int i = 0; i < nPulseIdxs; i++ )
        {
            int idx = m_pnCurrentColIdxs[pPulseIdxs[i]];
            if(strcmp(&m_pszCurrentLine[idx], &m_pszLastLine[idx]) != 0)
                return false;
        }
        return true;
    }

    // copy the data into a record from the given indices and using the struct defn
    void copyDataToRecord(int *pIdxs, int nIdxs, SpylidarFieldDefn *pDefn, char *pRecord)
    {
        for( int i = 0; i < nIdxs; i++ )
        {
            int idx = m_pnCurrentColIdxs[pIdxs[i]];
            char *pszString = &m_pszCurrentLine[idx];
            SpylidarFieldDefn *pElDefn = &pDefn[i];
            if( pElDefn->cKind == 'i' )
            {
#if defined (_MSC_VER) && _MSC_VER < 1900
                // early versions of MSVC don't have strtoll
                __int64 data = _strtoi64(pszString, NULL, 10);
                #define PRINTF_IFMT "%I64d"
#else
                long long data = strtoll(pszString, NULL, 10);
                #define PRINTF_IFMT "%lld"
#endif            
                switch(pElDefn->nSize)
                {
                    case 1:
                    {
                        if( (data < NPY_MIN_INT8) || (data > NPY_MAX_INT8))
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_IFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_int8 d = (npy_int8)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 2:
                    {
                        if( (data < NPY_MIN_INT16) || (data > NPY_MAX_INT16))
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_IFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_int16 d = (npy_int16)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 4:
                    {
                        if( (data < NPY_MIN_INT32) || (data > NPY_MAX_INT32))
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_IFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_int32 d = (npy_int32)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 8:
                    {
                        if( (data < NPY_MIN_INT64) || (data > NPY_MAX_INT64))
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_IFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_int64 d = (npy_int64)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    default:
                        fprintf(stderr, "Undefined element size %d\n", pElDefn->nSize);
                        break;
                }
            }
            else if( pElDefn->cKind == 'u')
            {
#if defined( _MSC_VER) && _MSC_VER < 1900
                // early versions of MSVC don't have strtoull
                unsigned __int64 data = _strtoui64(pszString, NULL, 10);
                #define PRINTF_UFMT "%I64u"
#else
                unsigned long long data = strtoull(pszString, NULL, 10);
                #define PRINTF_UFMT "%llu"
#endif
                switch(pElDefn->nSize)
                {
                    case 1:
                    {
                        if( data > NPY_MAX_UINT8 )
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_UFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_uint8 d = (npy_uint8)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 2:
                    {
                        if( data > NPY_MAX_UINT16 )
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_UFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_uint16 d = (npy_uint16)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 4:
                    {
                        if( data > NPY_MAX_UINT32 )
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_UFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_uint32 d = (npy_uint32)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 8:
                    {
                        if( data > NPY_MAX_UINT64 )
                        {
                            // TODO: exception?
                            fprintf(stderr, "Column %s data outside range of type (" PRINTF_UFMT ")\n", pElDefn->pszName, data);
                        }
                        npy_uint64 d = (npy_uint64)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    default:
                        fprintf(stderr, "Undefined element size %d\n", pElDefn->nSize);
                        break;
                }
            }
            else if( pElDefn->cKind == 'f')
            {
                double data = atof(pszString);
                switch(pElDefn->nSize)
                {
                    case 4:
                    {
                        float d = (float)data;
                        memcpy(&pRecord[pElDefn->nOffset], &d, sizeof(d));
                        break;
                    }
                    case 8:
                    {
                        memcpy(&pRecord[pElDefn->nOffset], &data, sizeof(data));
                        break;
                    }
                    default:
                        fprintf(stderr, "Undefined element size %d\n", pElDefn->nSize);
                        break;
                }
            }
            else
            {
                fprintf(stderr, "Unknown kind code %c\n", pElDefn->cKind);
            }
        }
    }

private:
    bool m_bFirst;
    int m_nFields;
    char m_cCommentChar;
    char *m_pszCurrentLine;
    char *m_pszLastLine;
    int *m_pnCurrentColIdxs;
    int *m_pnLastColIdxs;
};

static PyObject *PyASCIIReader_readData(PyASCIIReader *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;
    PyObject *pTuple = NULL;

    try
    {
        CReadState state(self->nPulseFields + self->nPointFields, self->cCommentChar);

        Py_ssize_t nPulsesToIgnore = 0;
        if(nPulseStart < self->nPulsesRead)
        {
            // the start location before the current location
            // reset to beginning and read through
#ifdef HAVE_ZLIB
            if( self->gz_file != NULL )
            {
                gzseek(self->gz_file, 0, SEEK_SET);
            }
#endif
            if( self->unc_file != NULL )
            {
                rewind(self->unc_file);
            }
            nPulsesToIgnore = nPulseStart;
            self->nPulsesRead = 0;
            self->bFinished = false;
        }
        else if(nPulseStart > self->nPulsesRead)
        {
            nPulsesToIgnore = (nPulseStart - self->nPulsesRead);
        }

        const char *pszErrorString = NULL;

        while(nPulsesToIgnore > 0)
        {
            if(!state.getNewLine(self, &pszErrorString))
            {
                if( pszErrorString != NULL )
                {
                    PyErr_SetString(GETSTATE_FC->error, pszErrorString);
                    return NULL;
                }
                self->bFinished = true;
                break;
            }

            if( !self->bTimeSequential || !state.isSamePulse(self->pPulseLineIdxs, self->nPulseFields))
            {
                nPulsesToIgnore--;
                self->nPulsesRead++;
            }
        }

        // we take a few liberties at this point. 
        // we don't actually have a struct for these since they
        // are user defined. So we just use type char.
        // Because we set the size, this should be ok.
        pylidar::CVector<char> pulseVector(nInitSize, nGrowBy, 
                self->pPulseDefn[0].nStructTotalSize);
        pylidar::CVector<char> pointVector(nInitSize, nGrowBy, 
                self->pPointDefn[0].nStructTotalSize);

        char *pulseItem = new char[self->pPulseDefn[0].nStructTotalSize];
        char *pointItem = new char[self->pPointDefn[0].nStructTotalSize];

        // find offset of NUMBER_OF_RETURNS and PTS_START_IDX so we can fill them in
        int nNumOfReturnsOffset = -1;
        int nPtsStartIdxOffset = -1;
        int i = 0;
        while( self->pPulseDefn[i].pszName != NULL )
        {
            if( strcmp(self->pPulseDefn[i].pszName, "NUMBER_OF_RETURNS") == 0)
                nNumOfReturnsOffset = self->pPulseDefn[i].nOffset;
            else if( strcmp(self->pPulseDefn[i].pszName, "PTS_START_IDX") == 0)
                nPtsStartIdxOffset = self->pPulseDefn[i].nOffset;

            i++;
        }
        if( (nNumOfReturnsOffset == -1) || (nPtsStartIdxOffset == -1))
        {
            // this should never happen
            fprintf(stderr, "couldn't find NUMBER_OF_RETURNS or PTS_START_IDX");
        }

        while( nPulses > 0 )
        {
            if(!state.getNewLine(self, &pszErrorString))
            {
                if( pszErrorString != NULL )
                {
                    PyErr_SetString(GETSTATE_FC->error, pszErrorString);
                    delete[] pulseItem;
                    delete[] pointItem;
                    return NULL;
                }
                self->bFinished = true;
                break;
            }

            bool bSamePulse = false;
            if( self->bTimeSequential )
                bSamePulse = state.isSamePulse(self->pPulseLineIdxs, self->nPulseFields);

            if( !bSamePulse )
            {
                // new pulse
                if( self->bTimeSequential )
                    state.copyDataToRecord(self->pPulseLineIdxs, self->nPulseFields, self->pPulseDefn, pulseItem);

                // set PTS_START_IDX
                npy_uint64 nPtsStartIdx = pointVector.getNumElems();
                memcpy(&pulseItem[nPtsStartIdxOffset], &nPtsStartIdx, sizeof(nPtsStartIdx));
                npy_uint8 nNumReturns = 0;
                memcpy(&pulseItem[nNumOfReturnsOffset], &nNumReturns, sizeof(nNumReturns));

                pulseVector.push(pulseItem);
                nPulses--;
                self->nPulsesRead++;
            }


            // add our new point
            state.copyDataToRecord(self->pPointLineIdxs, self->nPointFields, self->pPointDefn, pointItem);
            pointVector.push(pointItem);

            // update NUMBER_OF_RETURNS on pulse
            char *pLastPulse = pulseVector.getLastElement();
            if( pLastPulse != NULL )
            {
                npy_uint8 nNumReturns;
                memcpy(&nNumReturns, &pLastPulse[nNumOfReturnsOffset], sizeof(nNumReturns));
                nNumReturns++;
                memcpy(&pLastPulse[nNumOfReturnsOffset], &nNumReturns, sizeof(nNumReturns));
            }
        }
        delete[] pulseItem;
        delete[] pointItem;

        PyArrayObject *pPulses = pulseVector.getNumpyArray(self->pPulseDefn);
        PyArrayObject *pPoints = pointVector.getNumpyArray(self->pPointDefn);

        // build tuple
        pTuple = PyTuple_Pack(2, pPulses, pPoints);

        Py_DECREF(pPulses);
        Py_DECREF(pPoints);
    }
    catch(std::bad_alloc& ba)
    {
        PyErr_SetString(GETSTATE_FC->error, "Out of memory");
        return NULL;
    }

    return pTuple;
}

/* Table of methods */
static PyMethodDef PyASCIIReader_methods[] = {
    {"readData", (PyCFunction)PyASCIIReader_readData, METH_VARARGS, 
        "reads data. pass pulsestart, pulseend"}, 
    {NULL}  /* Sentinel */
};

static PyObject *PyASCIIReader_getFinished(PyASCIIReader *self, void *closure)
{
    if( self->bFinished )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyASCIIReader_getPulsesRead(PyASCIIReader *self, void *closure)
{
    return PyLong_FromSsize_t(self->nPulsesRead);
}

/* get/set */
static PyGetSetDef PyASCIIReader_getseters[] = {
    {(char*)"finished", (getter)PyASCIIReader_getFinished, NULL, (char*)"Get Finished reading state", NULL}, 
    {(char*)"pulsesRead", (getter)PyASCIIReader_getPulsesRead, NULL, (char*)"Get number of pulses read", NULL},
    {NULL}  /* Sentinel */
};

// Return a dictionary of typeCode / names
static PyObject *GetFormatNameDict()
{
    PyObject *pFormatNameDict = PyDict_New();
    PyObject *pKey = PyLong_FromLong(ASCII_UNKNOWN);
#if PY_MAJOR_VERSION >= 3
    PyObject *pValue = PyUnicode_FromString("Unknown");
#else
    PyObject *pValue = PyString_FromString("Unknown");
#endif
    PyDict_SetItem(pFormatNameDict, pKey, pValue);
    Py_DECREF(pValue);

    pKey = PyLong_FromLong(ASCII_UNCOMPRESSED);
#if PY_MAJOR_VERSION >= 3
    pValue = PyUnicode_FromString("Uncompressed File");
#else
    pValue = PyString_FromString("Uncompressed File");
#endif
    PyDict_SetItem(pFormatNameDict, pKey, pValue);
    Py_DECREF(pValue);

    pKey = PyLong_FromLong(ASCII_GZIP);
#if PY_MAJOR_VERSION >= 3
    pValue = PyUnicode_FromString("GZip File");
#else
    pValue = PyString_FromString("GZip File");
#endif
    PyDict_SetItem(pFormatNameDict, pKey, pValue);
    Py_DECREF(pValue);

    return pFormatNameDict;
}

static PyTypeObject PyASCIIReaderType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_ascii.Reader",         /*tp_name*/
    sizeof(PyASCIIReader),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyASCIIReader_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "ASCII File Read object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyASCIIReader_methods,             /* tp_methods */
    0,             /* tp_members */
    PyASCIIReader_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyASCIIReader_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};


#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__ascii(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_ascii(void)
#endif
{
    PyObject *pModule;
    struct ASCIIState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_ascii", module_methods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_ascii.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

    /* ascii file read type */
    PyASCIIReaderType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyASCIIReaderType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyASCIIReaderType);
    PyModule_AddObject(pModule, "Reader", (PyObject *)&PyASCIIReaderType);

    // dictionary of format names
    PyObject *pFormatNameDict = GetFormatNameDict();
    PyModule_AddObject(pModule, "FORMAT_NAMES", pFormatNameDict);

    // format presence flags
#ifdef HAVE_ZLIB
    Py_INCREF(Py_True);
    PyModule_AddObject(pModule, "HAVE_ZLIB", Py_True);
#else
    Py_INCREF(Py_False);
    PyModule_AddObject(pModule, "HAVE_ZLIB", Py_False);
#endif

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
