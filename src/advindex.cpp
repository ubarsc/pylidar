/*
 * advindex.cpp
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
#include <spatialindex/capi/sidx_api.h>
#include "pylvector.h"

/* An exception object for this module */
/* created in the init function */
struct AdvIndexState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct AdvIndexState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct AdvIndexState _state;
#endif

// structure for array returned from getPoints
typedef struct {
    double x;
    double y;
    npy_int64 idx;
} SPointInfo;

/* field info for pylidar_structArrayToNumpy */
static SpylidarFieldDefn PointInfoFields[] = {
    CREATE_FIELD_DEFN(SPointInfo, x, 'f'),
    CREATE_FIELD_DEFN(SPointInfo, y, 'f'),
    CREATE_FIELD_DEFN(SPointInfo, idx, 'i'),
    {NULL} // Sentinel
};

#if PY_MAJOR_VERSION >= 3
static int advindex_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int advindex_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_advindex",
        NULL,
        sizeof(struct AdvIndexState),
        NULL,
        NULL,
        advindex_traverse,
        advindex_clear,
        NULL
};
#endif

/* Python object wrapping a libspatialindex IndexH */
typedef struct 
{
    PyObject_HEAD

    IndexH idx;

} PyAdvIndex;

/* init method - open index */
static int 
PyAdvIndex_init(PyAdvIndex *self, PyObject *args, PyObject *kwds)
{
    const char *pszFname = NULL;
    int nIdxType, nNewFile;

    if( !PyArg_ParseTuple(args, "sii", &pszFname, &nIdxType, &nNewFile) )
    {
        return -1;
    }

    // set up the index
    IndexPropertyH props = IndexProperty_Create();

    // TODO: maybe check nIdxType is one of the INDEX_* constants?
    IndexProperty_SetIndexType(props, (RTIndexType)nIdxType);
    IndexProperty_SetIndexStorage(props, RT_Disk);
    IndexProperty_SetFileName(props, pszFname);
    IndexProperty_SetOverwrite(props, nNewFile);

    self->idx = Index_Create(props);

    // finished with the properties
    IndexProperty_Destroy(props);

    if( (self->idx == NULL ) || !Index_IsValid(self->idx) )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Unable to create Index");
        return -1;
    }

    return 0;
}

/* destructor - close and delete */
static void 
PyAdvIndex_dealloc(PyAdvIndex *self)
{
    if( self->idx != NULL )
    {
        Index_Flush(self->idx);
        Index_Destroy(self->idx);
    }
}

static PyObject *PyAdvIndex_setPoints(PyAdvIndex *self, PyObject *args)
{
    PyArrayObject *pXArray, *pYArray, *pIDArray;
    if( !PyArg_ParseTuple(args, "OOO:setPoints", &pXArray, &pYArray, &pIDArray) )
    {
        return NULL;
    }

    // all numpy arrays?
    if( !PyArray_Check(pXArray) || !PyArray_Check(pYArray) || !PyArray_Check(pIDArray) )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "All arguments must be numpy arrays");
        return NULL;
    }

    // check types and size
    if( (PyArray_TYPE(pXArray) != NPY_FLOAT64) ||
        (PyArray_TYPE(pYArray) != NPY_FLOAT64) )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Both Coordinate arrays must be float64");
        return NULL;
    }

    if( PyArray_TYPE(pIDArray) != NPY_UINT64 )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "ID array must be uint64");
        return NULL;
    }

    if( (PyArray_NDIM(pXArray) != 1) || (PyArray_NDIM(pYArray) != 1) ||
        (PyArray_NDIM(pIDArray) != 1) )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "All arrays must be 1d");
        return NULL;
    }

    npy_intp nSize = PyArray_DIM(pXArray, 0);
    if( (nSize != PyArray_DIM(pYArray, 0)) ||
        (nSize != PyArray_DIM(pIDArray, 0)) )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "All arrays must be the same size");
        return NULL;
    }

    double dMinArray[2];
    double dMaxArray[2];
    double dX, dY;
    npy_uint64 nID;
    for( npy_intp n = 0; n < nSize; n++ )
    {
        dX = *(double*)PyArray_GETPTR1(pXArray, n);
        dY = *(double*)PyArray_GETPTR1(pYArray, n);
        nID = *(npy_uint64*)PyArray_GETPTR1(pIDArray, n);

        dMinArray[0] = dX;
        dMaxArray[0] = dX;
        dMinArray[1] = dY;
        dMaxArray[1] = dY;

        if( Index_InsertData(self->idx, nID, dMinArray, dMaxArray, 2, NULL, 0) != RT_None )
        {
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_Format(GETSTATE(m)->error, "Error adding point to index: %f %f %s", dX, dY,
                    Error_GetLastErrorMsg());
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

static PyObject *PyAdvIndex_getPoints(PyAdvIndex *self, PyObject *args)
{
    double dXMin, dYMin, dXMax, dYMax;
    if( !PyArg_ParseTuple(args, "dddd:getPoints", &dXMin, &dYMin, &dXMax, &dYMax) )
    {
        return NULL;
    }

    double dMinArray[2];
    double dMaxArray[2];
    dMinArray[0] = dXMin;
    dMaxArray[0] = dXMax;
    dMinArray[1] = dYMin;
    dMaxArray[1] = dYMax;

    IndexItemH *pItems = NULL;
    uint64_t nResults = 0;
    if( Index_Intersects_obj(self->idx, dMinArray, dMaxArray, 2, &pItems, &nResults) != RT_None )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_Format(GETSTATE(m)->error, "Error querying index: %s", Error_GetLastErrorMsg());
        return NULL;
    }

    // create new array and copy coords and ids in
    pylidar::CVector<SPointInfo> infoArray(nResults, 0);
    for( uint64_t i = 0; i < nResults; i++ )
    {
        double *pdmin, *pdmax;
        uint32_t d;
        IndexItem_GetBounds(pItems[i], &pdmin, &pdmax, &d);
        int64_t dx = IndexItem_GetID(pItems[i]);

        SPointInfo *pItem = infoArray.getElem(i);

        // just a point, so dmin == dmax
        pItem->x = pdmin[0];
        pItem->y = pdmin[1];
        pItem->idx = dx;

        Index_Free(pdmin);
        Index_Free(pdmax);
    }
    
    Index_DestroyObjResults(pItems, nResults);

    return (PyObject*)infoArray.getNumpyArray(PointInfoFields);
}

static PyObject *PyAdvIndex_getExtent(PyAdvIndex *self, PyObject *args)
{
    double *pdMin, *pdMax;
    uint32_t nDimension;

    Index_Flush(self->idx);

    if( Index_GetBounds(self->idx, &pdMin, &pdMax, &nDimension) != RT_None )
    {
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_Format(GETSTATE(m)->error, "Error querying index: %s", Error_GetLastErrorMsg());
        return NULL;
    }

    PyObject *pXMin = PyFloat_FromDouble(pdMin[0]);
    PyObject *pYMin = PyFloat_FromDouble(pdMin[1]);
    PyObject *pXMax = PyFloat_FromDouble(pdMax[0]);
    PyObject *pYMax = PyFloat_FromDouble(pdMax[1]);

    PyObject *pTuple = PyTuple_Pack(4, pXMin, pYMin, pXMax, pYMax);

    Index_Free(pdMin);
    Index_Free(pdMax);

    return pTuple;
}

/* Table of methods */
static PyMethodDef PyAdvIndex_methods[] = {
    {"setPoints", (PyCFunction)PyAdvIndex_setPoints, METH_VARARGS, 
            "Sets points into the spatial index, pass xarray, yarray, idarray"}, 
    {"getPoints", (PyCFunction)PyAdvIndex_getPoints, METH_VARARGS, 
            "Gets id of points within bounds, pass xmin, ymin, xmax, ymax"}, 
    {"getExtent", (PyCFunction)PyAdvIndex_getExtent, METH_NOARGS,
            "Gets the bounds of the spatial index as xmin, ymin, xmax, ymax"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyAdvIndexType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_advindex.Index",         /*tp_name*/
    sizeof(PyAdvIndex),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyAdvIndex_dealloc, /*tp_dealloc*/
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
    "Advances Spatial Indexing object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyAdvIndex_methods,             /* tp_methods */
    0,             /* tp_members */
    0,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyAdvIndex_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};


#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__advindex(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_advindex(void)
#endif
{
    PyObject *pModule;
    struct AdvIndexState *state;

    /* initialize the numpy stuff */
    import_array();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_advindex", NULL);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_advindex.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

    /* advanced index type */
    PyAdvIndexType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyAdvIndexType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyAdvIndexType);
    PyModule_AddObject(pModule, "Index", (PyObject *)&PyAdvIndexType);

    // libspatialindex types
    PyModule_AddIntConstant(pModule, "INDEX_RTREE", RT_RTree);
    PyModule_AddIntConstant(pModule, "INDEX_MVRTREE", RT_MVRTree);
    PyModule_AddIntConstant(pModule, "INDEX_TPRTREE", RT_TPRTree);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
