/*
 * pulsewaves.cpp
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
#include "pylvector.h"

#include "pulsereader.hpp"
#include "pulsewriter.hpp"

/* An exception object for this module */
/* created in the init function */
struct PulseWavesState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct PulseWavesState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct PulseWavesState _state;
#endif

typedef struct {
    npy_int64 time;
    npy_int64 offset;
    double x_idx;
    double y_idx;
    double x_origin;
    double y_origin;
    double z_origin;
    double x_target;
    double y_target;
    double z_target;
    npy_int16 first_returning_sample;
    npy_int16 last_returning_sample;
    npy_uint16 descriptor_index;
    npy_uint8 intensity;
} SPulseWavesPulse;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn PulseWavesPulseFields[] = {
    CREATE_FIELD_DEFN(SPulseWavesPulse, time, 'i'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, offset, 'i'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, x_idx, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, y_idx, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, x_origin, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, y_origin, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, z_origin, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, x_target, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, y_target, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, z_target, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, first_returning_sample, 'i'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, last_returning_sample, 'i'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, descriptor_index, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, intensity, 'u'),
    {NULL} // Sentinel
};

/* Structure for points */
typedef struct {
    double x;
    double y;
    double z;
    npy_uint8 classification;
} SPulseWavesPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn PulseWavesPointFields[] = {
    CREATE_FIELD_DEFN(SPulseWavesPoint, x, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPoint, y, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPoint, z, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesPoint, classification, 'u'),
    {NULL} // Sentinel
};

/* Structure for waveform Info */
typedef struct {
    npy_uint32 number_of_waveform_received_bins;
    npy_uint64 received_start_idx;
    npy_uint32 number_of_waveform_transmitted_bins;
    npy_uint64 transmitted_start_idx;
    double      range_to_waveform_start;
} SPulseWavesWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn PulseWavesWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, number_of_waveform_transmitted_bins, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, transmitted_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, range_to_waveform_start, 'f'),
    {NULL} // Sentinel
};

// Python object wrapping a PULSEreader
typedef struct {
    PyObject_HEAD
    PULSEreader *pReader;
    bool bFinished;
} PyPulseWavesFileRead;

#if PY_MAJOR_VERSION >= 3
static int pulsewaves_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int pulsewaves_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_pulsewaves",
        NULL,
        sizeof(struct PulseWavesState),
        NULL,
        NULL,
        pulsewaves_traverse,
        pulsewaves_clear,
        NULL
};
#endif

/* destructor - close and delete */
static void 
PyPulseWavesFileRead_dealloc(PyPulseWavesFileRead *self)
{
    if( self->pReader != NULL )
    {
        self->pReader->close();
        delete self->pReader;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* init method - open file */
static int 
PyPulseWavesFileRead_init(PyPulseWavesFileRead *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname) )
    {
        return -1;
    }

    self->bFinished = false;

    PULSEreadOpener pulsereadopener;
    pulsereadopener.set_file_name(pszFname);

    self->pReader = pulsereadopener.open();
    if( self->pReader == NULL )
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Unable to open pulsewaves file");
        return -1;
    }

    return 0;
}

static PyObject *PyPulseWavesFileRead_readData(PyPulseWavesFileRead *self, PyObject *args)
{

    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyPulseWavesFileRead_methods[] = {
    {"readHeader", (PyCFunction)PyPulseWavesFileRead_readData, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyPulseWavesFileReadType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_pulsewaves.FileRead",         /*tp_name*/
    sizeof(PyPulseWavesFileRead),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyPulseWavesFileRead_dealloc, /*tp_dealloc*/
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
    "PulseWaves File Reader object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyPulseWavesFileRead_methods,             /* tp_methods */
    0,             /* tp_members */
    0,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyPulseWavesFileRead_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__pulsewaves(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_pulsewaves(void)
#endif
{
    PyObject *pModule;
    struct PulseWavesState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_pulsewaves", module_methods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_pulsewaves.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

    /* PulseWaves file read type */
    PyPulseWavesFileReadType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyPulseWavesFileReadType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyPulseWavesFileReadType);
    PyModule_AddObject(pModule, "PulseWavesFileRead", (PyObject *)&PyPulseWavesFileReadType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}

