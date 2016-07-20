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
#include <ctype.h>
#include <new>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "pylvector.h"

#ifdef HAVE_ZLIB
    #include "zlib.h"
#endif

// for CVector
static const int nGrowBy = 1000;
static const int nInitSize = 40000;

static const int nMaxLineSize = 8192;

/* An exception object for this module */
/* created in the init function */
struct ASCIIState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct ASCIIState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
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

    const char *pszLastExt = &pszFileName[i];

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
            (strcmp(pszLastExt, ".csv") == 0 ) ) )
    {
        // just check first char is a digit
        if( isdigit(aData[0]) )
        {
            nType = ASCII_UNCOMPRESSED;
        }
    }

    // insert other tests here

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

    uint64_t nPulsesRead;

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
SpylidarFieldDefn *DTypeListToFieldDef(PyObject *pList, int *pnFields, int **ppnIdxs,
        PyObject *error)
{
    if( !PySequence_Check(pList) )
    {
        PyErr_SetString(error, "Parameter is not a sequence");
        return NULL;
    }

    Py_ssize_t nSize = PySequence_Size(pList);
    // create Defn
    SpylidarFieldDefn *pDefn = (SpylidarFieldDefn*)calloc(nSize, sizeof(SpylidarFieldDefn));
    // idxs
    *ppnIdxs = (int*)calloc(nSize, sizeof(int));

    int nOffset = 0;
    for( Py_ssize_t i = 0; i < nSize; i++ )
    {
        PyObject *pElem = PySequence_GetItem(pList, i);
        if( !PySequence_Check(pElem) || ( PySequence_Size(pElem) != 3 ) )
        {
            PyErr_SetString(error, "Each element must be a 3 element sequence");
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

        PyObject *pNumpyDType = PySequence_GetItem(pElem, 1);
        // we can't actually do much with numpy.uint16 etc
        // which is what is passed in. Turn into numpy descr for more info
        PyArray_Descr *pDescr = NULL;
        if( !PyArray_DescrConverter(pNumpyDType, &pDescr) )
        {
            PyErr_SetString(error, "Couldn't convert 2nd element type to numpy Descr");
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

        // now for the idxs
        PyObject *pIdx = PySequence_GetItem(pElem, 2);
        if( !PyLong_Check(pIdx) )
        {
            PyErr_SetString(error, "3rd element must be int");
            FreeDefn(pDefn, nSize);
            free(*ppnIdxs);
            *ppnIdxs = NULL;
            return NULL;
        }

        (*ppnIdxs)[i] = PyLong_AsLong(pIdx);
    }

    // now do nStructTotalSize
    for( Py_ssize_t i = 0; i < nSize; i++ )
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
    const char *pszFname = NULL;
    int nType;
    PyObject *pPulseDTypeList, *pPointDTypeList;

    if( !PyArg_ParseTuple(args, "siOO", &pszFname, &nType, &pPulseDTypeList,
                &pPointDTypeList ) )
    {
        return -1;
    }

    PyObject *m;
#if PY_MAJOR_VERSION >= 3
    // best way I could find for obtaining module reference
    // from inside a class method. Not needed for Python < 3.
    m = PyState_FindModule(&moduledef);
#endif
    PyObject *error = GETSTATE(m)->error;

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
    self->pPulseLineIdxs = NULL;
    self->pPointLineIdxs = NULL;

    // create our definitions
    self->pPulseDefn = DTypeListToFieldDef(pPulseDTypeList, &self->nPulseFields, 
            &self->pPulseLineIdxs, error);
    if( self->pPulseDefn == NULL )
    {
        // error should be set
        return -1;
    }

    self->pPointDefn = DTypeListToFieldDef(pPointDTypeList, &self->nPointFields, 
            &self->pPointLineIdxs, error);
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
    CReadState()
    {
        m_pszBuffer1 = (char*)malloc(nMaxLineSize * sizeof(char));
        if( m_pszBuffer1 == NULL )
        {
            throw std::bad_alloc();
        }
        m_pszBuffer2 = (char*)malloc(nMaxLineSize * sizeof(char));
        if( m_pszBuffer2 == NULL )
        {
            free(m_pszBuffer1);
            throw std::bad_alloc();
        }
        m_pszCurrentLine = m_pszBuffer1;
        m_pszLastLine = m_pszBuffer2;
    }
    ~CReadState()
    {
        free(m_pszBuffer1);
        free(m_pszBuffer2);
    }

    bool getNewLine(PyASCIIReader *self)
    {
        // read into last line then clobber
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

        char *pszOldCurrent = m_pszCurrentLine;
        m_pszCurrentLine = m_pszLastLine;
        m_pszLastLine = pszOldCurrent;
    }

private:
    char *m_pszBuffer1;
    char *m_pszBuffer2;
    char *m_pszCurrentLine;
    char *m_pszLastLine;
};

static PyObject *PyASCIIReader_readData(PyASCIIReader *self, PyObject *args)
{
    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyASCIIReader_methods[] = {
    {"readData", (PyCFunction)PyASCIIReader_readData, METH_VARARGS, NULL}, 
    {NULL}  /* Sentinel */
};

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
    0,           /* tp_getset */
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

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
