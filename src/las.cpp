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
    PyObject_HEAD
    LASreader *pReader;
    bool bBuildPulses;
    bool bFinished;
} PyLasFile;

static const char *SupportedDriverOptions[] = {"BUILD_PULSES", NULL};
static PyObject *las_getSupportedOptions(PyObject *self, PyObject *args)
{
    // how many do we have?
    Py_ssize_t n;
    for( n = 0; SupportedDriverOptions[n] != NULL; n++ )
    {
        // do nothing
    }

    // now do it for real
    PyObject *pTuple = PyTuple_New(n);
    for( n = 0; SupportedDriverOptions[n] != NULL; n++ )
    {
        PyObject *pStr;
        const char *psz = SupportedDriverOptions[n];
#if PY_MAJOR_VERSION >= 3
        pStr = PyUnicode_FromString(psz);
#else
        pStr = PyString_FromString(psz);
#endif
        PyTuple_SetItem(pTuple, n, pStr);
    }

    return pTuple;
}

// module methods
static PyMethodDef module_methods[] = {
    {"getSupportedOptions", (PyCFunction)las_getSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options"},
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

/* destructor - close and delete */
static void 
PyLasFile_dealloc(PyLasFile *self)
{
    if(self->pReader != NULL)
    {
        self->pReader->close();
        delete self->pReader;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* init method - open file */
static int 
PyLasFile_init(PyLasFile *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;
PyObject *pOptionDict;

    if( !PyArg_ParseTuple(args, "sO", &pszFname, &pOptionDict ) )
    {
        return -1;
    }

    if( !PyDict_Check(pOptionDict) )
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Last parameter to init function must be a dictionary");
        return -1;
    }

    self->bFinished = false;
    self->bBuildPulses = true;

    /* Check creation options */
    PyObject *pBuildPulses = PyDict_GetItemString(pOptionDict, "BUILD_PULSES");
    if( pBuildPulses != NULL )
    {
        if( PyBool_Check(pBuildPulses) )
        {
            self->bBuildPulses = (pBuildPulses == Py_True);
        }
        else
        {
            // raise Python exception
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_SetString(GETSTATE(m)->error, "BUILD_PULSES must be true or false");    
            return -1;
        }
    }

    LASreadOpener lasreadopener;
    lasreadopener.set_file_name(pszFname);
    self->pReader = lasreadopener.open();

    if( self->pReader == NULL )
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Unable to open las file");
        return -1;
    }

    return 0;
}

/* calculate the length in case they change in future */
#define GET_LENGTH(x) (sizeof(x) / sizeof(x[0]))

static PyObject *PyLasFile_readHeader(PyLasFile *self, PyObject *args)
{
    PyObject *pHeaderDict = PyDict_New();
    LASheader *pHeader = &self->pReader->header;

#if PY_MAJOR_VERSION >= 3
    PyObject *pVal = PyUnicode_FromStringAndSize(pHeader->file_signature, 
                GET_LENGTH(pHeader->file_signature));
#else
    PyObject *pVal = PyString_FromStringAndSize(pHeader->file_signature, 
                GET_LENGTH(pHeader->file_signature));
#endif
    PyDict_SetItemString(pHeaderDict, "file_signature", pVal);

    pVal = PyLong_FromLong(pHeader->file_source_ID);
    PyDict_SetItemString(pHeaderDict, "file_source_ID", pVal);

    pVal = PyLong_FromLong(pHeader->global_encoding);
    PyDict_SetItemString(pHeaderDict, "global_encoding", pVal);

    pVal = PyLong_FromLong(pHeader->project_ID_GUID_data_1);
    PyDict_SetItemString(pHeaderDict, "project_ID_GUID_data_1", pVal);

    pVal = PyLong_FromLong(pHeader->project_ID_GUID_data_2);
    PyDict_SetItemString(pHeaderDict, "project_ID_GUID_data_2", pVal);

    pVal = PyLong_FromLong(pHeader->project_ID_GUID_data_3);
    PyDict_SetItemString(pHeaderDict, "project_ID_GUID_data_3", pVal);

    pylidar::CVector<U8> project_ID_GUID_data_4Vector(pHeader->project_ID_GUID_data_4, 
                            sizeof(pHeader->project_ID_GUID_data_4));    
    pVal = project_ID_GUID_data_4Vector.getNumpyArray(NPY_UINT8);
    PyDict_SetItemString(pHeaderDict, "project_ID_GUID_data_4", pVal);

    pVal = PyLong_FromLong(pHeader->version_major);
    PyDict_SetItemString(pHeaderDict, "version_major", pVal);

    pVal = PyLong_FromLong(pHeader->version_minor);
    PyDict_SetItemString(pHeaderDict, "version_minor", pVal);

#if PY_MAJOR_VERSION >= 3
    pVal = PyUnicode_FromStringAndSize(pHeader->system_identifier, 
                GET_LENGTH(pHeader->system_identifier));
#else
    pVal = PyString_FromStringAndSize(pHeader->system_identifier, 
                GET_LENGTH(pHeader->system_identifier));
#endif
    PyDict_SetItemString(pHeaderDict, "system_identifier", pVal);

#if PY_MAJOR_VERSION >= 3
    pVal = PyUnicode_FromStringAndSize(pHeader->generating_software, 
                GET_LENGTH(pHeader->generating_software));
#else
    pVal = PyString_FromStringAndSize(pHeader->generating_software, 
                GET_LENGTH(pHeader->generating_software));
#endif
    PyDict_SetItemString(pHeaderDict, "generating_software", pVal);

    pVal = PyLong_FromLong(pHeader->file_creation_day);
    PyDict_SetItemString(pHeaderDict, "file_creation_day", pVal);

    pVal = PyLong_FromLong(pHeader->file_creation_year);
    PyDict_SetItemString(pHeaderDict, "file_creation_year", pVal);

    pVal = PyLong_FromLong(pHeader->header_size);
    PyDict_SetItemString(pHeaderDict, "header_size", pVal);

    pVal = PyLong_FromLong(pHeader->offset_to_point_data);
    PyDict_SetItemString(pHeaderDict, "offset_to_point_data", pVal);

    pVal = PyLong_FromLong(pHeader->number_of_variable_length_records);
    PyDict_SetItemString(pHeaderDict, "number_of_variable_length_records", pVal);

    pVal = PyLong_FromLong(pHeader->point_data_format);
    PyDict_SetItemString(pHeaderDict, "point_data_format", pVal);

    pVal = PyLong_FromLong(pHeader->point_data_record_length);
    PyDict_SetItemString(pHeaderDict, "point_data_record_length", pVal);

    pVal = PyLong_FromLong(pHeader->number_of_point_records);
    PyDict_SetItemString(pHeaderDict, "number_of_point_records", pVal);

    pylidar::CVector<U32> number_of_points_by_returnVector(pHeader->number_of_points_by_return, 
                            sizeof(pHeader->number_of_points_by_return));    
    pVal = number_of_points_by_returnVector.getNumpyArray(NPY_UINT32);
    PyDict_SetItemString(pHeaderDict, "number_of_points_by_return", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_x);
    PyDict_SetItemString(pHeaderDict, "max_x", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_x);
    PyDict_SetItemString(pHeaderDict, "min_x", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_y);
    PyDict_SetItemString(pHeaderDict, "max_y", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_y);
    PyDict_SetItemString(pHeaderDict, "min_y", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_z);
    PyDict_SetItemString(pHeaderDict, "max_z", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_z);
    PyDict_SetItemString(pHeaderDict, "min_z", pVal);

    return pHeaderDict;
}

static PyObject *PyLasFile_readData(PyLasFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses, nCount;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    for( nCount = 0; nCount < nPulses; nCount++ )
    {
        if( !self->pReader->read_point() )
        {
            self->bFinished = true;
            break;
        }
    }

    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyLasFile_methods[] = {
    {"readHeader", (PyCFunction)PyLasFile_readHeader, METH_NOARGS, NULL},
    {"readData", (PyCFunction)PyLasFile_readData, METH_VARARGS, NULL}, 
    {NULL}  /* Sentinel */
};

static PyObject *PyLasFile_getBuildPulses(PyLasFile *self, void *closure)
{
    if( self->bBuildPulses )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyLasFile_getHasSpatialIndex(PyLasFile *self, void *closure)
{
    if( self->pReader->get_index() != NULL )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyLasFile_getFinished(PyLasFile *self, void *closure)
{
    if( self->bFinished )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

/* get/set */
static PyGetSetDef PyLasFile_getseters[] = {
    {"build_pulses", (getter)PyLasFile_getBuildPulses, NULL, 
        "Whether we are building pulses of multiple points when reading", NULL},
    {"hasSpatialIndex", (getter)PyLasFile_getHasSpatialIndex, NULL,
        "Whether a spatial index exists for this file", NULL},
    {"finished", (getter)PyLasFile_getFinished, NULL, 
        "Whether we have finished reading the file or not", NULL},
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
