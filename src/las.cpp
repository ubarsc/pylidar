/*
 * las.cpp
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

#include "lasreader.hpp"

/* An exception object for this module */
/* created in the init function */
struct LasState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct LasState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct LasState _state;
#endif

/* Structure for LAS points */
typedef struct {
    double x;
    double y;
    double z;
    npy_uint16 intensity;
    npy_uint8 return_number;
    npy_uint8 number_of_returns;
    npy_uint8 scan_direction_flag;
    npy_uint8 edge_of_flight_line;
    npy_uint8 classification;
    npy_uint8 synthetic_flag;
    npy_uint8 keypoint_flag;
    npy_uint8 withheld_flag;
    npy_int8 scan_angle_rank;
    npy_uint8 user_data;
    npy_uint16 point_source_ID;
    double gps_time;
    npy_uint16 red;
    npy_uint16 green;
    npy_uint16 blue;
    npy_uint16 alpha;
} SLasPoint;

/* field info for pylidar_structArrayToNumpy */
static SpylidarFieldDefn LasPointFields[] = {
    CREATE_FIELD_DEFN(SLasPoint, x, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, y, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, z, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, intensity, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, return_number, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, scan_direction_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, edge_of_flight_line, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, classification, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, synthetic_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, keypoint_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, withheld_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, scan_angle_rank, 'i'),
    CREATE_FIELD_DEFN(SLasPoint, user_data, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, point_source_ID, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, gps_time, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, red, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, green, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, blue, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, alpha, 'u'),
    {NULL} // Sentinel
};

/* Python object wrapping a LASreader */
typedef struct {
    LASreader *pReader;

} PyLasFile;

/* destructor - close and delete tc */
static void 
PyLasFile_dealloc(PyLasFile *self)
{
    if(self->pReader != NULL)
    {
        self->pReader->close();
        delete self->pReader;
    }
}

/* init method - open file */
static int 
PyLasFile_init(PyLasFile *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname ) )
    {
        return -1;
    }

    return 0;
}

/* Table of methods */
static PyMethodDef PyLasFile_methods[] = {
    {NULL}  /* Sentinel */
};

/* get/set */
static PyGetSetDef PyLasFile_getseters[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLasFileType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_las.File",         /*tp_name*/
    sizeof(PyLasFile),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyLasFile_dealloc, /*tp_dealloc*/
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
    "Las File object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyLasFile_methods,             /* tp_methods */
    0,             /* tp_members */
    PyLasFile_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyLasFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

// module methods
static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static int las_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int las_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_las",
        NULL,
        sizeof(struct LasState),
        module_methods,
        NULL,
        las_traverse,
        las_clear,
        NULL
};
#endif

#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__las(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_las(void)
#endif
{
    PyObject *pModule;
    struct LasState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_las", module_methods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_las.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }

    /* Scan file type */
    PyLasFileType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyLasFileType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyLasFileType);
    PyModule_AddObject(pModule, "LasFile", (PyObject *)&PyLasFileType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
