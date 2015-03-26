/*
 *  cgalinterp.cpp
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

//#include <math>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/algorithm.h>
#include <CGAL/Origin.h>
#include <CGAL/squared_distance_2.h>

#include <Python.h>
#include "numpy/arrayobject.h"
    
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT                                         CGALCoordType;
typedef K::Vector_2                                   CGALVector;
typedef K::Point_2                                    CGALPoint;
    
typedef CGAL::Delaunay_triangulation_2<K>             DelaunayTriangulation;
typedef CGAL::Interpolation_traits_2<K>               InterpTraits;
typedef CGAL::Delaunay_triangulation_2<K>::Vertex_handle    Vertex_handle;
typedef CGAL::Delaunay_triangulation_2<K>::Face_handle    Face_handle;
    
typedef std::vector< std::pair<CGALPoint, CGALCoordType> >   CoordinateVector;
typedef std::map<CGALPoint, CGALCoordType, K::Less_xy_2>     PointValueMap;

/* An exception object for this module */
/* created in the init function */
struct CGALInterpState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct CGALInterpState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct CGALInterpState _state;
#endif

static PyObject *cgalinterp_naturalneighbour(PyObject *self, PyObject *args)
{
PyObject *pXVals, *pYVals, *pZVals, *pXGrid, *pYGrid;
PyObject *pOutArray;

    if( !PyArg_ParseTuple(args, "OOOOO:NaturalNeighbour", &pPythonFeature, &pBBoxObject, &nXSize, &nYSize, &nLineWidth))
        return NULL;

    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXGrid) || !PyArray_Check(pYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }

    // TODO: check dimensions match
    // and types ok

    npy_intp nRows = PyArray_DIM(pXVals, 0);
    npy_intp nCols = PyArray_DIM(pXVals, 1);
    
    npy_intp nVals = PyArray_DIM(pXVals, 0);

    // Create output
    pOutArray = PyArray_EMPTY(2, PyArray_DIMS(pXGrid), NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }

    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        return NULL;
    }    


    if( PyArray_DIM(pXVals, 0) < 100 )
    {
        // check that these small number of points aren't all within a line
        double meanX = 0;
        double meanY = 0;
                
        double varX = 0;
        double varY = 0;
                
        for(size_t i = 0; i < nVals; ++i)
        {
            meanX += *((double*)PyArray_GETPTR1(pXVals, i))
            meanY += *((double*)PyArray_GETPTR1(pYVals, i))
        }
                
        meanX = meanX / nVals;
        meanY = meanY / nVals;
                
        //std::cout << "meanX = " << meanX << std::endl;
        //std::cout << "meanY = " << meanY << std::endl;
                
        for(size_t i = 0; i < nVals; ++i)
        {
            varX += *((double*)PyArray_GETPTR1(pXVals, i)) - meanX;
            varY += *((double*)PyArray_GETPTR1(pYVals, i)) - meanY;
        }
                
        varX = fabs(varX / nVals);
        varY = fabs(varY / nVals);
                
        //std::cout << "varX = " << varX << std::endl;
        //std::cout << "varY = " << varX << std::endl;
                
        if((varX < 4) || (varY < 4))
        {
            PyErr_SetString(GETSTATE(self)->error, "Points are all within a line.");
            return NULL;
        }

        try
        {
            DelaunayTriangulation *dt = new DelaunayTriangulation();
            PointValueMap *values = new PointValueMap();
            
            for(npy_intp i = 0; i < nVals; ++i)
            {
                K::Point_2 cgalPt(*((double*)PyArray_GETPTR1(pXVals, i)), *((double*)PyArray_GETPTR1(pYVals, i)));
                dt->insert(cgalPt);
                CGALCoordType value = *((double*)PyArray_GETPTR1(pZVals, i));
                values->insert(std::make_pair(cgalPt, value));
            }


            for(npy_intp i  = 0; i < nRows; ++i)
            {
                for(npy_intp j = 0; j < nCols; ++j)
                {
                    
                    //K::Point_2 p(xGrid[i][j], yGrid[i][j]);
                    K::Point_2 p(*((double*)PyArray_GETPTR2(pXVals, i, j)), 
                                *((double*)PyArray_GETPTR2(pYVals, i, j)));
                    CoordinateVector coords;
                    CGAL::Triple<std::back_insert_iterator<CoordinateVector>, K::FT, bool> result = CGAL::natural_neighbor_coordinates_2(*dt, p, std::back_inserter(coords));
                    if(!result.third)
                    {
                        // Not within convex hull of dataset
                        *((double*)PyArray_GETPTR2(pOutArray, i, j) = 0.0;
                    }
                    else
                    {
                        CGALCoordType norm = result.second;
                        CGALCoordType outValue = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, CGAL::Data_access<PointValueMap>(*this->values));
                        *((double*)PyArray_GETPTR2(pOutArray, i, j) = outValue;
                    }
                }
            }
            
            delete dt;
            delete values;
        }
        catch(std::exception &e)
        {
            PyErr_SetString(GETSTATE(self)->error, e.str());
            return NULL;
        }
        return pOutArray;
    }
    /*
    double** interpGridNN(double *xVals, double *yVals, double *zVals, size_t nVals, double **xGrid, double **yGrid, size_t nXGrid, size_t nYGrid)throw(std::exception)
    {
        double **outVals = NULL;
        try
        {
            if(nVals < 3)
            {
                throw std::exception("Not enough points, need at least 3.");
            }
            else if(nVals < 100)
            {
                double meanX = 0;
                double meanY = 0;
                
                double varX = 0;
                double varY = 0;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    meanX += xVals[i];
                    meanY += yVals[i];
                }
                
                meanX = meanX / nVals;
                meanY = meanY / nVals;
                
                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    varX += xVals[i] - meanX;
                    varY += yVals[i] - meanY;
                }
                
                varX = fabs(varX / nVals);
                varY = fabs(varY / nVals);
                
                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;
                
                if((varX < 4) | (varY < 4))
                {
                    throw std::exception("Points are all within a line.");
                }
            }
            
            DelaunayTriangulation *dt = new DelaunayTriangulation();
            PointValueMap *values = new PointValueMap();
            
            for(size_t i = 0; i < nVals; ++i)
            {
                K::Point_2 cgalPt(xVals[i], yVals[i]);
                dt->insert(cgalPt);
                CGALCoordType value = zVals[i];
                values->insert(std::make_pair(cgalPt, value));
            }
            
            outVals = new double*[nYGrid];
            for(size_t i  = 0; i < nYGrid; ++i)
            {
                outVals[i] = new double*[nXGrid];
                for(size_t j = 0; j < nXGrid; ++j)
                {
                    
                    K::Point_2 p(xGrid[i][j], yGrid[i][j]);
                    CoordinateVector coords;
                    CGAL::Triple<std::back_insert_iterator<CoordinateVector>, K::FT, bool> result = CGAL::natural_neighbor_coordinates_2(*dt, p, std::back_inserter(coords));
                    if(!result.third)
                    {
                        // Not within convex hull of dataset
                        outVals[i][j] = 0.0;
                    }
                    else
                    {
                        CGALCoordType norm = result.second;
                        CGALCoordType outValue = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, CGAL::Data_access<PointValueMap>(*this->values));
                        outVals[i][j] = outValue;
                    }
                }
            }
            
            delete dt;
            delete values;
        }
        catch(std::exception &e)
        {
            throw e;
        }
        return outVals;
    }
    

    double** interpGridPlaneFit(double *xVals, double *yVals, double *zVals, size_t nVals, double **xGrid, double **yGrid, size_t nXGrid, size_t nYGrid)throw(std::exception)
    {
        double **outVals = NULL;
        try
        {
            /*
            if(nVals < 3)
            {
                throw std::exception("Not enough points, need at least 3.");
            }
            else if(nVals < 100)
            {
                double meanX = 0;
                double meanY = 0;
                
                double varX = 0;
                double varY = 0;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    meanX += xVals[i];
                    meanY += yVals[i];
                }
                
                meanX = meanX / nVals;
                meanY = meanY / nVals;
                
                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;
                
                for(size_t i = 0; i < nVals; ++i)
                {
                    varX += xVals[i] - meanX;
                    varY += yVals[i] - meanY;
                }
                
                varX = fabs(varX / nVals);
                varY = fabs(varY / nVals);
                
                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;
                
                if((varX < 4) | (varY < 4))
                {
                    throw std::exception("Points are all within a line.");
                }
            }
            
            DelaunayTriangulation *dt = new DelaunayTriangulation();
            PointValueMap *values = new PointValueMap();
            
            for(size_t i = 0; i < nVals; ++i)
            {
                K::Point_2 cgalPt(xVals[i], yVals[i]);
                dt->insert(cgalPt);
                CGALCoordType value = zVals[i];
                values->insert(std::make_pair(cgalPt, value));
            }
            
            outVals = new double*[nYGrid];
            for(size_t i  = 0; i < nYGrid; ++i)
            {
                outVals[i] = new double*[nXGrid];
                for(size_t j = 0; j < nXGrid; ++j)
                {
                    outVals[i][j] = 0.0;
                }
            }
            
            delete dt;
            delete values;
        }
        catch(std::exception &e)
        {
            throw e;
        }
        return outVals;
    }
    */
    

/* Our list of functions in this module*/
static PyMethodDef CGALInterpMethods[] = {
    {"NaturalNeighbour", cgalinterp_naturalneighbour, METH_VARARGS,
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
    {NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int cgalinterp_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int cgalinterp_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cgalinterp",
        NULL,
        sizeof(struct CGALInterpState),
        CGALInterpMethods,
        NULL,
        cgalinterp_traverse,
        cgalinterp_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit_cgalinterp(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initcgalinterp(void)
#endif
{
    PyObject *pModule;
    struct CGALInterpState *state;

    /* initialize the numpy stuff */
    import_array();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("cgalinterp", CGALInterpMethods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("cgal.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}


