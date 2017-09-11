/*
 *  performNNInterp.c
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


#include <math.h>

#include <Python.h>
#include "numpy/arrayobject.h"

#include "nn.h"

/* An exception object for this module */
/* created in the init function */
struct PyNNInterpState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct PyNNInterpState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct PyNNInterpState _state;
#endif

static PyObject *pynninterp_naturalneighbour(PyObject *self, PyObject *args)
{
    //std::cout.precision(12);
    PyArrayObject *pXVals, *pYVals, *pZVals, *pXGrid, *pYGrid;
    PyArrayObject *pOutArray;
    npy_intp nRows, nCols, nVals, i, j, nPtsOutGrid, idx;
    point *inPts, *gPts;
    
    if( !PyArg_ParseTuple(args, "OOOOO:NaturalNeighbour", &pXVals, &pYVals, &pZVals, &pXGrid, &pYGrid))
        return NULL;
    
    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXGrid) || !PyArray_Check(pYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }

    // check dims
    if( (PyArray_NDIM(pXVals) != 1) || (PyArray_NDIM(pYVals) != 1) || (PyArray_NDIM(pZVals) != 1) || 
            (PyArray_NDIM(pXGrid) != 2) || (PyArray_NDIM(pYGrid) != 2) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Arrays should be 1d, 1d, 1d, 2d and 2d respectively");
        return NULL;
    }
    
    // Check dimensions match
    if( (PyArray_DIM(pXVals, 0) != PyArray_DIM(pYVals, 0)) | (PyArray_DIM(pXVals, 0) != PyArray_DIM(pZVals, 0)))
    {
        PyErr_SetString(GETSTATE(self)->error, "Training X, Y and Z arrays must all be of the same length");
        return NULL;
    }
    
    if( (PyArray_DIM(pXGrid, 0) != PyArray_DIM(pYGrid, 0)) | (PyArray_DIM(pXGrid, 1) != PyArray_DIM(pYGrid, 1)))
    {
        PyErr_SetString(GETSTATE(self)->error, "X and Y grids must have the same dimensions");
        return NULL;
    }
    
    // check types ok
    if( (PyArray_TYPE(pXVals) != NPY_DOUBLE) || (PyArray_TYPE(pYVals) != NPY_DOUBLE) || 
        (PyArray_TYPE(pZVals) != NPY_DOUBLE) || (PyArray_TYPE(pXGrid) != NPY_DOUBLE) ||
        (PyArray_TYPE(pYGrid) != NPY_DOUBLE) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All input arrays must be double");
        return NULL;
    }
    
    nRows = PyArray_DIM(pXGrid, 0);
    nCols = PyArray_DIM(pXGrid, 1);
    
    nVals = PyArray_DIM(pXVals, 0);
    
    // Create output
    pOutArray = (PyArrayObject*)PyArray_EMPTY(2, PyArray_DIMS(pXGrid), NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }
    
    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        Py_DECREF(pOutArray);
        return NULL;
    }
    
    // BUILD POINT ARRAYS
    inPts = malloc(nVals * sizeof(point));
    if( inPts == NULL )
    {
        Py_DECREF(pOutArray);
        PyErr_SetString(GETSTATE(self)->error, "Failed to create temporary array");
        return NULL;
    }
    for(i = 0; i < nVals; ++i)
    {
        inPts[i].x = *((double*)PyArray_GETPTR1(pXVals, i));
        inPts[i].y = *((double*)PyArray_GETPTR1(pYVals, i));
        inPts[i].z = *((double*)PyArray_GETPTR1(pZVals, i));
    }
    
    nPtsOutGrid = nRows * nCols;
    idx = 0;
    gPts = malloc(nPtsOutGrid * sizeof(point));
    for(i = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            gPts[idx].x = *((double*)PyArray_GETPTR2(pXGrid, i, j));
            gPts[idx].y = *((double*)PyArray_GETPTR2(pYGrid, i, j));
            gPts[idx].z = 0.0;
            ++idx;
        }
    }
    
    nnpi_interpolate_points(nVals, inPts, 0.0, nPtsOutGrid, gPts);
    
    free(inPts);
    
    
    // POPULATE GRID
    idx = 0;
    for(i  = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            *((double*)PyArray_GETPTR2(pOutArray, i, j)) = gPts[idx++].z;
        }
    }
    
    free(gPts);
    
    return (PyObject*)pOutArray;
}

static PyObject *pynninterp_naturalneighbour_pts(PyObject *self, PyObject *args)
{
    //std::cout.precision(12);
    PyArrayObject *pXVals, *pYVals, *pZVals, *pXYGrid;
    PyArrayObject *pOutArray;
    npy_intp nPts, nVals, i;
    point *inPts, *gPts;
    
    if( !PyArg_ParseTuple(args, "OOOO:NaturalNeighbourPts", &pXVals, &pYVals, &pZVals, &pXYGrid))
        return NULL;
    
    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }

    // check dims
    if( (PyArray_NDIM(pXVals) != 1) || (PyArray_NDIM(pYVals) != 1) || (PyArray_NDIM(pZVals) != 1) || 
            (PyArray_NDIM(pXYGrid) != 2) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Arrays should be 1d, 1d, 1d and 2d respectively");
        return NULL;
    }
    
    // Check dimensions match
    if( (PyArray_DIM(pXVals, 0) != PyArray_DIM(pYVals, 0)) | (PyArray_DIM(pXVals, 0) != PyArray_DIM(pZVals, 0)))
    {
        PyErr_SetString(GETSTATE(self)->error, "Training X, Y and Z arrays must all be of the same length");
        return NULL;
    }
    
    if( PyArray_DIM(pXYGrid, 1) != 2 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Interpolation point array must be shape N*2");
        return NULL;
    }
    
    // check types ok
    if( (PyArray_TYPE(pXVals) != NPY_DOUBLE) || (PyArray_TYPE(pYVals) != NPY_DOUBLE) || 
        (PyArray_TYPE(pZVals) != NPY_DOUBLE) || (PyArray_TYPE(pXYGrid) != NPY_DOUBLE) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All input arrays must be double");
        return NULL;
    }
    
    nPts = PyArray_DIM(pXYGrid, 0);
    nVals = PyArray_DIM(pXVals, 0);
    
    // Create output
    pOutArray = (PyArrayObject*)PyArray_EMPTY(1, &nPts, NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }
    
    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        Py_DECREF(pOutArray);
        return NULL;
    }
    
    // BUILD POINT ARRAYS
    inPts = malloc(nVals * sizeof(point));
    if( inPts == NULL )
    {
        Py_DECREF(pOutArray);
        PyErr_SetString(GETSTATE(self)->error, "Failed to create temporary array");
        return NULL;
    }
    for(i = 0; i < nVals; ++i)
    {
        inPts[i].x = *((double*)PyArray_GETPTR1(pXVals, i));
        inPts[i].y = *((double*)PyArray_GETPTR1(pYVals, i));
        inPts[i].z = *((double*)PyArray_GETPTR1(pZVals, i));
    }
    
    gPts = malloc(nPts * sizeof(point));
    for(i = 0; i < nPts; ++i)
    {
        gPts[i].x = *((double*)PyArray_GETPTR2(pXYGrid, i, 0));
        gPts[i].y = *((double*)PyArray_GETPTR2(pXYGrid, i, 1));
        gPts[i].z = 0.0;
    }
    
    nnpi_interpolate_points(nVals, inPts, 0.0, nPts, gPts);
    
    free(inPts);
    
    // POPULATE POINTS
    for(i  = 0; i < nPts; ++i)
    {
        *((double*)PyArray_GETPTR1(pOutArray, i)) = gPts[i].z;
    }
    
    free(gPts);
    
    return (PyObject*)pOutArray;
}



static PyObject *pynninterp_linear(PyObject *self, PyObject *args)
{
    //std::cout.precision(12);
    PyArrayObject *pXVals, *pYVals, *pZVals, *pXGrid, *pYGrid;
    PyArrayObject *pOutArray;
    npy_intp nRows, nCols, nVals, i, j, nPtsOutGrid, idx;
    point *inPts, *gPts;
    
    if( !PyArg_ParseTuple(args, "OOOOO:Linear", &pXVals, &pYVals, &pZVals, &pXGrid, &pYGrid))
        return NULL;
    
    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXGrid) || !PyArray_Check(pYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }
    
    // check dims
    if( (PyArray_NDIM(pXVals) != 1) || (PyArray_NDIM(pYVals) != 1) || (PyArray_NDIM(pZVals) != 1) || 
            (PyArray_NDIM(pXGrid) != 2) || (PyArray_NDIM(pYGrid) != 2) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Arrays should be 1d, 1d, 1d, 2d and 2d respectively");
        return NULL;
    }

    // Check dimensions match
    if( (PyArray_DIM(pXVals, 0) != PyArray_DIM(pYVals, 0)) | (PyArray_DIM(pXVals, 0) != PyArray_DIM(pZVals, 0)))
    {
        PyErr_SetString(GETSTATE(self)->error, "Training X, Y and Z arrays must all be of the same length");
        return NULL;
    }
    
    if( (PyArray_DIM(pXGrid, 0) != PyArray_DIM(pYGrid, 0)) | (PyArray_DIM(pXGrid, 1) != PyArray_DIM(pYGrid, 1)))
    {
        PyErr_SetString(GETSTATE(self)->error, "X and Y grids must have the same dimensions");
        return NULL;
    }
    
    // check types ok
    if( (PyArray_TYPE(pXVals) != NPY_DOUBLE) || (PyArray_TYPE(pYVals) != NPY_DOUBLE) || 
        (PyArray_TYPE(pZVals) != NPY_DOUBLE) || (PyArray_TYPE(pXGrid) != NPY_DOUBLE) ||
        (PyArray_TYPE(pYGrid) != NPY_DOUBLE) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All input arrays must be double");
        return NULL;
    }
    
    nRows = PyArray_DIM(pXGrid, 0);
    nCols = PyArray_DIM(pXGrid, 1);
    
    nVals = PyArray_DIM(pXVals, 0);
    
    // Create output
    pOutArray = (PyArrayObject*)PyArray_EMPTY(2, PyArray_DIMS(pXGrid), NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }
    
    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        Py_DECREF(pOutArray);
        return NULL;
    }
    
    // BUILD POINT ARRAYS
    inPts = malloc(nVals * sizeof(point));
    if( inPts == NULL )
    {
        Py_DECREF(pOutArray);
        PyErr_SetString(GETSTATE(self)->error, "Failed to create temporary array");
        return NULL;
    }
    for(i = 0; i < nVals; ++i)
    {
        inPts[i].x = *((double*)PyArray_GETPTR1(pXVals, i));
        inPts[i].y = *((double*)PyArray_GETPTR1(pYVals, i));
        inPts[i].z = *((double*)PyArray_GETPTR1(pZVals, i));
    }
    
    nPtsOutGrid = nRows * nCols;
    idx = 0;
    gPts = malloc(nPtsOutGrid * sizeof(point));
    for(i  = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            gPts[idx].x = *((double*)PyArray_GETPTR2(pXGrid, i, j));
            gPts[idx].y = *((double*)PyArray_GETPTR2(pYGrid, i, j));
            gPts[idx].z = 0.0;
            ++idx;
        }
    }
    
    lpi_interpolate_points(nVals, inPts, nPtsOutGrid, gPts);
    
    free(inPts);
    
    
    // POPULATE GRID
    idx = 0;
    for(i  = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            *((double*)PyArray_GETPTR2(pOutArray, i, j)) = gPts[idx++].z;
        }
    }
    
    free(gPts);
    
    return (PyObject*)pOutArray;
}

static PyObject *pynninterp_linear_pts(PyObject *self, PyObject *args)
{
    //std::cout.precision(12);
    PyArrayObject *pXVals, *pYVals, *pZVals, *pXYGrid;
    PyArrayObject *pOutArray;
    npy_intp nPts, nVals, i;
    point *inPts, *gPts;
    
    if( !PyArg_ParseTuple(args, "OOOO:NaturalNeighbourPts", &pXVals, &pYVals, &pZVals, &pXYGrid))
        return NULL;
    
    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }

    // check dims
    if( (PyArray_NDIM(pXVals) != 1) || (PyArray_NDIM(pYVals) != 1) || (PyArray_NDIM(pZVals) != 1) || 
            (PyArray_NDIM(pXYGrid) != 2) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Arrays should be 1d, 1d, 1d and 2d respectively");
        return NULL;
    }
    
    // Check dimensions match
    if( (PyArray_DIM(pXVals, 0) != PyArray_DIM(pYVals, 0)) | (PyArray_DIM(pXVals, 0) != PyArray_DIM(pZVals, 0)))
    {
        PyErr_SetString(GETSTATE(self)->error, "Training X, Y and Z arrays must all be of the same length");
        return NULL;
    }
    
    if( PyArray_DIM(pXYGrid, 1) != 2 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Interpolation point array must be shape N*2");
        return NULL;
    }
    
    // check types ok
    if( (PyArray_TYPE(pXVals) != NPY_DOUBLE) || (PyArray_TYPE(pYVals) != NPY_DOUBLE) || 
        (PyArray_TYPE(pZVals) != NPY_DOUBLE) || (PyArray_TYPE(pXYGrid) != NPY_DOUBLE) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All input arrays must be double");
        return NULL;
    }
    
    nPts = PyArray_DIM(pXYGrid, 0);
    nVals = PyArray_DIM(pXVals, 0);
    
    // Create output
    pOutArray = (PyArrayObject*)PyArray_EMPTY(1, &nPts, NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }
    
    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        Py_DECREF(pOutArray);
        return NULL;
    }
    
    // BUILD POINT ARRAYS
    inPts = malloc(nVals * sizeof(point));
    if( inPts == NULL )
    {
        Py_DECREF(pOutArray);
        PyErr_SetString(GETSTATE(self)->error, "Failed to create temporary array");
        return NULL;
    }
    for(i = 0; i < nVals; ++i)
    {
        inPts[i].x = *((double*)PyArray_GETPTR1(pXVals, i));
        inPts[i].y = *((double*)PyArray_GETPTR1(pYVals, i));
        inPts[i].z = *((double*)PyArray_GETPTR1(pZVals, i));
    }
    
    gPts = malloc(nPts * sizeof(point));
    for(i = 0; i < nPts; ++i)
    {
        gPts[i].x = *((double*)PyArray_GETPTR2(pXYGrid, i, 0));
        gPts[i].y = *((double*)PyArray_GETPTR2(pXYGrid, i, 1));
        gPts[i].z = 0.0;
    }
    
    lpi_interpolate_points(nVals, inPts, nPts, gPts);
    
    free(inPts);
    
    // POPULATE POINTS
    for(i  = 0; i < nPts; ++i)
    {
        *((double*)PyArray_GETPTR1(pOutArray, i)) = gPts[i].z;
    }
    
    free(gPts);
    
    return (PyObject*)pOutArray;
}


/* Our list of functions in this module*/
static PyMethodDef PyNNInterpMethods[] = {
    {"NaturalNeighbour", pynninterp_naturalneighbour, METH_VARARGS,
        "Perform Natural Neighbour Interpolation\n"
        "call signature: arr = NaturalNeighbour(xvals, yvals, zvals, xgrid, ygrid)\n"
        "where:\n"
        "  xvals is a 1d array of the x values of the points\n"
        "  yvals is a 1d array of the y values of the points\n"
        "  zvals is a 1d array of the z values of the points\n"
        "xvals, yvals and zvals should have the same length\n"
        "  xgrid is a 2d array of x coordinates to interpolate at\n"
        "  ygrid is a 2d array of y coordinates to interpolate at\n"
        "xgrid and xgrid must be the same shape"},
    {"Linear", pynninterp_linear, METH_VARARGS,
        "Perform Linear (TIN) Interpolation\n"
        "call signature: arr = Linear(xvals, yvals, zvals, xgrid, ygrid)\n"
        "where:\n"
        "  xvals is a 1d array of the x values of the points\n"
        "  yvals is a 1d array of the y values of the points\n"
        "  zvals is a 1d array of the z values of the points\n"
        "xvals, yvals and zvals should have the same length\n"
        "  xgrid is a 2d array of x coordinates to interpolate at\n"
        "  ygrid is a 2d array of y coordinates to interpolate at\n"
        "xgrid and xgrid must be the same shape"},
    {"NaturalNeighbourPts", pynninterp_naturalneighbour_pts, METH_VARARGS,
        "Perform Natural Neighbour Interpolation at points\n"
        "call signature: arr = NaturalNeighbourPts(xvals, yvals, zvals, xygrid)\n"
        "where:\n"
        "  xvals is a 1d array of the x values of the points\n"
        "  yvals is a 1d array of the y values of the points\n"
        "  zvals is a 1d array of the z values of the points\n"
        "xvals, yvals and zvals should have the same length\n"
        "  xygrid is a 2d array of coordinates in the form x,y\n"},
    {"LinearPts", pynninterp_linear_pts, METH_VARARGS,
        "Perform Linear (TIN) Interpolation at points\n"
        "call signature: arr = LinearPts(xvals, yvals, zvals, xygrid)\n"
        "where:\n"
        "  xvals is a 1d array of the x values of the points\n"
        "  yvals is a 1d array of the y values of the points\n"
        "  zvals is a 1d array of the z values of the points\n"
        "xvals, yvals and zvals should have the same length\n"
        "  xygrid is a 2d array of coordinates in the form x,y\n"},
    {NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int pynninterp_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int pynninterp_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "nninterp",
    NULL,
    sizeof(struct PyNNInterpState),
    PyNNInterpMethods,
    NULL,
    pynninterp_traverse,
    pynninterp_clear,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_nninterp(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initnninterp(void)
#endif
{
    PyObject *pModule;
    struct PyNNInterpState *state;
    
    /* initialize the numpy stuff */
    import_array();
    
#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("nninterp", PyNNInterpMethods);
#endif
    if( pModule == NULL )
        INITERROR;
    
    state = GETSTATE(pModule);
    
    /* Create and add our exception type */
    state->error = PyErr_NewException("nninterp.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    
#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
