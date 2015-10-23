/*
 * riegl.cpp
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
#include "pylidar.h"

#include <riegl/scanlib.hpp>

/* An exception object for this module */
/* created in the init function */
struct RieglState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct RieglState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct RieglState _state;
#endif

/* Python object wrapping a scanlib::basic_rconnection */
typedef struct
{
    PyObject_HEAD
    std::shared_ptr<scanlib::basic_rconnection> rc;
    Py_ssize_t nCurrPulse;
} PyRieglScanFile;

/* destructor - close and delete tc */
static void 
PyRieglScanFile_dealloc(PyRieglScanFile *self)
{
    self->rc->close();
    self->rc.reset();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* init method - open file */
static int 
PyRieglScanFile_init(PyRieglScanFile *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname ) )
    {
        return -1;
    }

    /* Raises exception on failure? */
    self->rc = scanlib::basic_rconnection::create(pszFname);

    self->nCurrPulse = 0;
    return 0;
}

/* Structure for pulses */
typedef struct {
    uint64_t pulseID;
    uint64_t gpsTime;
    float azimuth;
    float zenith;
    uint32_t scanline;
    uint16_t scanlineIdx;
    double xIdx;
    double yIdx;
    uint16_t sourceId;
    double xOrigin;
    double yOrigin;
    float zOrigin;
    uint32_t pointStartIdx;
    uint16_t pointCount;
} SRieglPulse;

/* field info for pylidar_structArrayToNumpy */
static SpylidarFieldDefn RieglPulseFields[] = {
    CREATE_FIELD_DEFN(SRieglPulse, pulseID, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, gpsTime, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, azimuth, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, zenith, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, scanline, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, scanlineIdx, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, yIdx, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, xIdx, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, sourceId, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, xOrigin, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, yOrigin, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, zOrigin, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, pointStartIdx, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, pointCount, 'u'),
    {NULL} // Sentinel
};

/* Structure for points */
typedef struct {
    uint64_t returnId;
    uint64_t gpsTime;
    float amplitudeReturn;
    float widthReturn;
    uint8_t classification;
    double x;
    double y;
    float z;
} SRieglPoint;

/* field info for pylidar_structArrayToNumpy */
static SpylidarFieldDefn RieglPointFields[] = {
    CREATE_FIELD_DEFN(SRieglPoint, returnId, 'u'),
    CREATE_FIELD_DEFN(SRieglPoint, gpsTime, 'u'),
    CREATE_FIELD_DEFN(SRieglPoint, amplitudeReturn, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, widthReturn, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, classification, 'u'),
    CREATE_FIELD_DEFN(SRieglPoint, x, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, y, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, z, 'f'),
    {NULL} // Sentinel
};

static PyObject *PyRieglScanFile_readData(PyRieglScanFile *self, PyObject *args)
{
Py_ssize_t nPulseStart, nPulseEnd;

    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyRieglScanFile_methods[] = {
    {"readData", (PyCFunction)PyRieglScanFile_readData, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyRieglScanFileType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_riegl.ScanFile",         /*tp_name*/
    sizeof(PyRieglScanFile),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyRieglScanFile_dealloc, /*tp_dealloc*/
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
    "Riegl Scan File object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyRieglScanFile_methods,             /* tp_methods */
    0,             /* tp_members */
    0,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyRieglScanFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};
#if PY_MAJOR_VERSION >= 3

static int riegl_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int riegl_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_riegl",
        NULL,
        sizeof(struct RieglState),
        NULL,
        NULL,
        riegl_traverse,
        riegl_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__riegl(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_riegl(void)
#endif
{
    PyObject *pModule;
    struct RieglState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_riegl", RieglMethods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_riegl.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }

    /* Scan file type */
    PyRieglScanFileType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyRieglScanFileType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyRieglScanFileType);
    PyModule_AddObject(pModule, "ScanFile", (PyObject *)&PyRieglScanFileType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
