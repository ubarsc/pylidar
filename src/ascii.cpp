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
#include <Python.h>
#include "numpy/arrayobject.h"
#include "pylvector.h"

#include "zlib.h"

// for CVector
static const int nGrowBy = 1000;
static const int nInitSize = 40000;

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

// file compression types - currently only gzip
#define ASCII_UNKNOWN 0
#define ASCII_GZIP 1

static PyObject *ascii_getFileType(PyObject *self, PyObject *args)
{
    const char *pszFileName;
    if( !PyArg_ParseTuple(args, "s:getFileType", &pszFileName))
        return NULL;

    int nLen = strlen(pszFileName);
    int nType = ASCII_UNKNOWN;
    FILE *pFH = fopen(pszFileName, "rb");
    if( pFH == NULL )
    {
        PyErr_Format(GETSTATE(self)->error, "Unable to open file: %s", pszFileName);
        return NULL;
    }

    if( ( nLen >= 3 ) && (pszFileName[nLen-1] == 'z') && (pszFileName[nLen-2] == 'g') &&
            (pszFileName[nLen-3] == '.') )
    {
        unsigned char aData[2];
        // read the first 2 bytes and confirm gzip header
        if( fread(aData, 1, sizeof(aData), pFH) != sizeof(aData) )
        {
            PyErr_Format(GETSTATE(self)->error, "Cannot read file: %s", pszFileName);
            fclose(pFH);
            return NULL;
        }

        if( ( aData[0] == 0x1f ) && ( aData[1] == 0x8b ) )
        {
            nType = ASCII_GZIP;
        }
    }

    fclose(pFH);

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
    gzFile  gz_file;
    uint64_t nPulsesRead;
} PyASCIIReader;

/* init method - open file */
static int 
PyASCIIReader_init(PyASCIIReader *self, PyObject *args, PyObject *kwds)
{
const char *pszFname = NULL;
int nType;

    if( !PyArg_ParseTuple(args, "si", &pszFname, &nType ) )
    {
        return -1;
    }

    // nType should come from getFileType()
    if( nType != ASCII_GZIP )
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "ASCII Compression type not supported");
        return -1;
    }

    self->gz_file = gzopen(pszFname, "rb");
    if( self->gz_file == NULL )
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Unable to open file");
        return -1;
    }

    self->nPulsesRead = 0;

    return 0;
}

/* destructor - close and delete */
static void 
PyASCIIReader_dealloc(PyASCIIReader *self)
{
    if( self->gz_file != NULL )
    {
        gzclose(self->gz_file);
    }
}

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
    PyModule_AddObject(pModule, "ASCIIFileRead", (PyObject *)&PyASCIIReaderType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
