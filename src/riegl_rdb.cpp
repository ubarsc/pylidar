/*
 * riegl_rdb.cpp
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

#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "pylvector.h"

#include <riegl/rdb.h>

/* An exception object for this module */
/* created in the init function */
struct RieglRDBState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct RieglRDBState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
static struct RieglRDBState _state;
#endif

typedef struct
{
    PyObject_HEAD
    char *pszFilename; // so we can re-create the file obj if needed
    RDBContext *pContext;
    RDBPointcloud *pPointCloud;

} PyRieglRDBFile;

// module methods
static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static int rieglrdb_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int rieglrdb_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_rieglrdb",
        NULL,
        sizeof(struct RieglRDBState),
        module_methods,
        NULL,
        rieglrdb_traverse,
        rieglrdb_clear,
        NULL
};
#endif

static void 
PyRieglRDBFile_dealloc(PyRieglRDBFile *self)
{
    free(self->pszFilename);
    if( self->pPointCloud != NULL )
    {
        rdb_pointcloud_delete(self->pContext, &self->pPointCloud);
    }
    if( self->pContext != NULL )
    {
        rdb_context_delete(&self->pContext);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int 
PyRieglRDBFile_init(PyRieglRDBFile *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname ) )
    {
        return -1;
    }
    
    // get context
    // TODO: log level and log path
    rdb_context_new(&self->pContext, "", "");
    
    // check we are running against the same version of the library we were compiled
    // against
    RDBString pszVersionString;
    rdb_library_license(self->pContext, &pszVersionString);
    if( strcmp(pszVersionString, "RIEGL_RDB_VERSION") != 0 )
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC->error, "Mismatched libraries - RDB lib differs in version string. "
            "Was compiled against version %s. Now running with %s\n", 
            "RIEGL_RDB_VERSION", pszVersionString);
        return -1;
    }
    

    // take a copy of the filename so we can re
    // create the file pointer.
    self->pszFilename = strdup(pszFname);

    return 0;
}

static PyObject *PyRieglRDBFile_readData(PyRieglRDBFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    Py_RETURN_NONE;
}

static PyObject *PyRieglRDBFile_readWaveforms(PyRieglRDBFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd;
    if( !PyArg_ParseTuple(args, "nn:readWaveforms", &nPulseStart, &nPulseEnd ) )
        return NULL;

    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyRieglRDBFile_methods[] = {
    {"readData", (PyCFunction)PyRieglRDBFile_readData, METH_VARARGS, NULL},
    {"readWaveforms", (PyCFunction)PyRieglRDBFile_readWaveforms, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

/* get/set */
static PyGetSetDef PyRieglRDBFile_getseters[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject PyRieglRDBFileType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_rieglrdb.RDBFile",         /*tp_name*/
    sizeof(PyRieglRDBFile),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyRieglRDBFile_dealloc, /*tp_dealloc*/
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
    "Riegl RDB File object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyRieglRDBFile_methods,             /* tp_methods */
    0,             /* tp_members */
    PyRieglRDBFile_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyRieglRDBFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};
#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__rieglrdb(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_rieglrdb(void)
#endif
{
    PyObject *pModule;
    struct RieglRDBState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_rieglrdb", module_methods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_rieglrdb.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

    /* Scan file type */
    PyRieglRDBFileType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyRieglRDBFileType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyRieglRDBFileType);
    PyModule_AddObject(pModule, "RDBFile", (PyObject *)&PyRieglRDBFileType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
