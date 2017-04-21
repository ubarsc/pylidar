/*
 * insidepoly.c
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

/* Adapted from TuiView */

#include <Python.h>
#include "numpy/arrayobject.h"
#include <string.h>
#include <stdlib.h>

#include "ogr_api.h"

/* define WKB_BYTE_ORDER depending on endian setting
 from pyconfig.h */
#if WORDS_BIGENDIAN == 1
    #define WKB_BYTE_ORDER wkbXDR
#else
    #define WKB_BYTE_ORDER wkbNDR
#endif

/* do a memcpy rather than cast and access so we work on SPARC etc with aligned reads */
#define READ_WKB_VAL(n, p)  memcpy(&n, p, sizeof(n)); p += sizeof(n);

typedef struct 
{
    PyArrayObject *pMask;
    PyArrayObject *pXCoords;
    PyArrayObject *pYCoords;
    npy_intp nSize;
} VectorParserData;

static VectorParserData *VectorParser_create(PyArrayObject *pOutMask, PyArrayObject *pXCoords,
                            PyArrayObject *pYCoords, npy_intp nSize)
{
    VectorParserData *pData;
    pData = (VectorParserData*)malloc(sizeof(VectorParserData));
    pData->pMask = pOutMask;
    pData->pXCoords = pXCoords;
    pData->pYCoords = pYCoords;
    pData->nSize = nSize;

    return pData;
}

static void VectorParser_destroy(VectorParserData *pData)
{
    free(pData);
}

static unsigned char* VectorParser_processPoint(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    /* x and y */
    pWKB += sizeof(double);
    pWKB += sizeof(double);
    if(hasz)
    {
        pWKB += sizeof(double);
    }

    return pWKB;
}

static unsigned char* VectorParser_processLineString(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    GUInt32 nPoints, n;

    READ_WKB_VAL(nPoints, pWKB)
    for( n = 0; n < nPoints; n++ )
    {
        pWKB += sizeof(double);
        pWKB += sizeof(double);
        if(hasz)
        {
            pWKB += sizeof(double);
        }
    }
    return pWKB;
}

/* See http://alienryderflex.com/polygon/ */
void precalc_values(int polyCorners, double *polyX, double *polyY, double *constant, double *multiple)
{
  int i, j=polyCorners-1;

    for(i=0; i<polyCorners; i++) 
    {
        if(polyY[j]==polyY[i]) 
        {
            constant[i]=polyX[i];
            multiple[i]=0; 
        }
        else 
        {
            constant[i]=polyX[i]-(polyY[i]*polyX[j])/(polyY[j]-polyY[i])+(polyY[i]*polyX[i])/(polyY[j]-polyY[i]);
            multiple[i]=(polyX[j]-polyX[i])/(polyY[j]-polyY[i]); 
        }
        j=i; 
    }
}

int pointInPolygon(int polyCorners, double x, double y,double *polyX, 
        double *polyY, double *constant, double *multiple) 
{
    int   i, j=polyCorners-1;
    int  oddNodes=0;

    for (i=0; i<polyCorners; i++) 
    {
        if ((((polyY[i]< y) && (polyY[j]>=y))
            ||   ((polyY[j]< y) && (polyY[i]>=y))) )
        {
            oddNodes^=(y*multiple[i]+constant[i]<x); 
        }
        j=i; 
    }

    return oddNodes; 
}

/* same as processLineString, but closes ring */
static unsigned char* VectorParser_processLinearRing(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    GUInt32 nPoints, n;
    double dx, dy;
    double *pPolyX, *pPolyY, *pConstant, *pMultiple;
    npy_intp nArrayIdx;
    npy_bool b;

    READ_WKB_VAL(nPoints, pWKB)
    if( nPoints > 0 )
    {
        pPolyX = (double*)malloc(nPoints * sizeof(double));
        pPolyY = (double*)malloc(nPoints * sizeof(double));
        pConstant = (double*)malloc(nPoints * sizeof(double));
        pMultiple = (double*)malloc(nPoints * sizeof(double));

        for( n = 0; n < nPoints; n++ )
        {
            READ_WKB_VAL(dx, pWKB)
            READ_WKB_VAL(dy, pWKB)
            if(hasz)
            {
                pWKB += sizeof(double);
            }
            pPolyX[n] = dx;
            pPolyY[n] = dy;
        }

        precalc_values(nPoints, pPolyX, pPolyY, pConstant, pMultiple);

        /* OK check all the points */
        for( nArrayIdx = 0; nArrayIdx < pData->nSize; nArrayIdx++ )
        {
            /* existing value - don't check if already in one poly */
            b = *((npy_bool*)PyArray_GETPTR1(pData->pMask, nArrayIdx));
            if( !b )
            {
                dx = *((double*)PyArray_GETPTR1(pData->pXCoords, nArrayIdx));
                dy = *((double*)PyArray_GETPTR1(pData->pYCoords, nArrayIdx));
                if( pointInPolygon(nPoints, dx, dy, pPolyX, pPolyY,
                        pConstant, pMultiple) )
                {
                    *((npy_bool*)PyArray_GETPTR1(pData->pMask, nArrayIdx)) = 1;
                }
            }
        }

        free(pPolyX);
        free(pPolyY);
        free(pConstant);
        free(pMultiple);
    }
    return pWKB;
}

static unsigned char* VectorParser_processPolygon(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    GUInt32 nRings, n;

    READ_WKB_VAL(nRings, pWKB)
    for( n = 0; n < nRings; n++ )
    {
        pWKB = VectorParser_processLinearRing(pData, pWKB, hasz);
    }
    return pWKB;
}

static unsigned char* VectorParser_processMultiPoint(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    GUInt32 nPoints, n;

    READ_WKB_VAL(nPoints, pWKB)
    for( n = 0; n < nPoints; n++ )
    {
        pWKB++; /* ignore endian */
        pWKB += sizeof(GUInt32); /* ignore type (must be point) */
        pWKB = VectorParser_processPoint(pData, pWKB, hasz);
    }
    return pWKB;
}

static unsigned char* VectorParser_processMultiLineString(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    GUInt32 nLines, n;

    READ_WKB_VAL(nLines, pWKB)
    for( n = 0; n < nLines; n++ )
    {
        pWKB++; /* ignore endian */
        pWKB += sizeof(GUInt32); /* ignore type */
        pWKB = VectorParser_processLineString(pData, pWKB, hasz);
    }
    return pWKB;
}

static unsigned char* VectorParser_processMultiPolygon(VectorParserData *pData, unsigned char *pWKB, int hasz)
{
    GUInt32 nPolys, n;

    READ_WKB_VAL(nPolys, pWKB)
    for( n = 0; n < nPolys; n++ )
    {
        pWKB++; /* ignore endian */
        pWKB += sizeof(GUInt32); /* ignore type */
        pWKB = VectorParser_processPolygon(pData, pWKB, hasz);
    }
    return pWKB;
}

static unsigned char* VectorParser_processWKB(VectorParserData *pData, unsigned char *pCurrWKB);

static unsigned char* VectorParser_processGeometryCollection(VectorParserData *pData, unsigned char *pWKB)
{
    GUInt32 nGeoms, n;

    READ_WKB_VAL(nGeoms, pWKB)
    for( n = 0; n < nGeoms; n++ )
    {
        /* start again! */
        pWKB = VectorParser_processWKB(pData, pWKB);
    }
    return pWKB;
}

static unsigned char* VectorParser_processWKB(VectorParserData *pData, unsigned char *pCurrWKB)
{
    GUInt32 nType;

    /* ignore byte order (should be native) */
    pCurrWKB++;
    READ_WKB_VAL(nType, pCurrWKB)
    switch(nType)
    {
    case wkbPoint:
        pCurrWKB = VectorParser_processPoint(pData, pCurrWKB, 0);
        break;
    case wkbPoint25D:
        pCurrWKB = VectorParser_processPoint(pData, pCurrWKB, 1);
        break;
    case wkbLineString:
        pCurrWKB = VectorParser_processLineString(pData, pCurrWKB, 0);
        break;
    case wkbLineString25D:
        pCurrWKB = VectorParser_processLineString(pData, pCurrWKB, 1);
        break;
    case wkbPolygon:
        pCurrWKB = VectorParser_processPolygon(pData, pCurrWKB, 0);
        break;
    case wkbPolygon25D:
        pCurrWKB = VectorParser_processPolygon(pData, pCurrWKB, 1);
        break;
    case wkbMultiPoint:
        pCurrWKB = VectorParser_processMultiPoint(pData, pCurrWKB, 0);
        break;
    case wkbMultiPoint25D:
        pCurrWKB = VectorParser_processMultiPoint(pData, pCurrWKB, 1);
        break;
    case wkbMultiLineString:
        pCurrWKB = VectorParser_processMultiLineString(pData, pCurrWKB, 0);
        break;
    case wkbMultiLineString25D:
        pCurrWKB = VectorParser_processMultiLineString(pData, pCurrWKB, 1);
        break;
    case wkbMultiPolygon:
        pCurrWKB = VectorParser_processMultiPolygon(pData, pCurrWKB, 0);
        break;
    case wkbMultiPolygon25D:
        pCurrWKB = VectorParser_processMultiPolygon(pData, pCurrWKB, 1);
        break;
    case wkbGeometryCollection:
    case wkbGeometryCollection25D:
        pCurrWKB = VectorParser_processGeometryCollection(pData, pCurrWKB);
        break;
    case wkbNone:
        /* pure attribute records */
        break;
    default:
        fprintf( stderr, "Unknown WKB code %d\n", nType);
        break;
    }
    return pCurrWKB;
}

/* bit of a hack here - this is what a SWIG object looks
 like. It is only defined in the source file so we copy it here*/
typedef struct {
  PyObject_HEAD
  void *ptr;
  /* the rest aren't needed (and a hassle to define here)
  swig_type_info *ty;
  int own;
  PyObject *next;*/
} SwigPyObject;

/* in the ideal world I would use SWIG_ConvertPtr etc
 but to avoid the whole dependence on SWIG headers
 and needing to reach into the GDAL source to get types etc
 I happen to know the 'this' attribute is a pointer to 
 OGRLayerShadow which is actually a pointer
 to a SwigPyObject whose ptr field is a pointer to a
 OGRLayer. Phew. 
 Given a python object this function returns the underlying
 pointer. Returns NULL on failure (exception string already set)*/
void *getUnderlyingPtrFromSWIGPyObject(PyObject *pObj, PyObject *pException)
{
    PyObject *pThisAttr; /* the 'this' field */
    SwigPyObject *pSwigThisAttr;
    void *pUnderlying;

    pThisAttr = PyObject_GetAttrString(pObj, "this");
    if( pThisAttr == NULL )
    {
        PyErr_SetString(pException, "object does not appear to be a swig type");
        return NULL;
    }

    /* i think it is safe to do this since pObj is still around*/
    Py_DECREF(pThisAttr);

    /* convert this to a SwigPyObject*/
    pSwigThisAttr = (SwigPyObject*)pThisAttr;

    /* get the ptr field*/
    pUnderlying = pSwigThisAttr->ptr;
    if( pUnderlying == NULL )
    {
        PyErr_SetString(pException, "underlying object is NULL");
        return NULL;
    }

    return pUnderlying;
}

/* An exception object for this module */
/* created in the init function */
struct InsidePolyState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct InsidePolyState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct InsidePolyState _state;
#endif

static PyObject *insidepoly_insideLayer(PyObject *self, PyObject *args)
{
    PyObject *pPythonLayer; /* of type ogr.Layer*/
    PyArrayObject *pXCoords, *pYCoords; /* numpy arrays */
    void *pPtr;
    OGRLayerH hOGRLayer;
    PyArrayObject *pOutMask;
    npy_intp nValues;
    VectorParserData *pParser;
    OGRFeatureH hFeature;
    int nCurrWKBSize;
    unsigned char *pCurrWKB;
    OGRGeometryH hGeometry;
    int nNewWKBSize;
    unsigned char *pNewWKB;

    if( !PyArg_ParseTuple(args, "OOO:insideLayer", &pPythonLayer, 
            &pXCoords, &pYCoords))
        return NULL;

    pPtr = getUnderlyingPtrFromSWIGPyObject(pPythonLayer, GETSTATE(self)->error);
    if( pPtr == NULL )
        return NULL;
    hOGRLayer = (OGRLayerH)pPtr;

    /* some checks */
    if( !PyArray_Check(pXCoords) || !PyArray_Check(pYCoords) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Inputs must be numpy arrays" );
        return NULL;
    }

    if( (PyArray_NDIM(pXCoords) != 1) || (PyArray_NDIM(pYCoords) != 1))
    {
        PyErr_SetString(GETSTATE(self)->error, "Input arrays must be 1d" );
        return NULL;
    }

    if( PyArray_DIM(pXCoords, 0) != PyArray_DIM(pYCoords, 0) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Input arrays must be same size" );
        return NULL;
    }

    if( (PyArray_TYPE(pXCoords) != NPY_DOUBLE) || (PyArray_TYPE(pYCoords) != NPY_DOUBLE))
    {
        PyErr_SetString(GETSTATE(self)->error, "Input arrays must be float64" );
        return NULL;
    }

    nValues = PyArray_DIM(pXCoords, 0);

    /* create output array - 0 (False) initially */
    pOutMask = (PyArrayObject*)PyArray_ZEROS(1, &nValues, NPY_BOOL, 0);
    if( pOutMask == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Unable to allocate array" );
        return NULL;
    }

    /* set up the object that does the writing */
    pParser = VectorParser_create(pOutMask, pXCoords, pYCoords, nValues);
    
    /* TODO: set spatial filter? */

    OGR_L_ResetReading(hOGRLayer);

    nCurrWKBSize = 0;
    pCurrWKB = NULL;

    while( ( hFeature = OGR_L_GetNextFeature(hOGRLayer)) != NULL )
    {
        hGeometry = OGR_F_GetGeometryRef(hFeature);
        if( hGeometry != NULL )
        {
            /* how big a buffer do we need? Grow if needed */
            nNewWKBSize = OGR_G_WkbSize(hGeometry);
            if( nNewWKBSize > nCurrWKBSize )
            {
                pNewWKB = (unsigned char*)realloc(pCurrWKB, nNewWKBSize);
                if( pNewWKB == NULL )
                {
                    /* realloc failed - bail out */
                    /* according to man page original not freed */
                    free(pCurrWKB);
                    Py_DECREF(pOutMask);
                    OGR_F_Destroy(hFeature);
                    VectorParser_destroy(pParser);
                    PyErr_SetString(GETSTATE(self)->error, "memory allocation failed");
                    return NULL;
                }
                else
                {
                    pCurrWKB = pNewWKB;
                    nCurrWKBSize = nNewWKBSize;
                }
            }
            /* read it in */
            OGR_G_ExportToWkb(hGeometry, WKB_BYTE_ORDER, pCurrWKB);
            /* write it to array */
            VectorParser_processWKB(pParser, pCurrWKB);
        }

        OGR_F_Destroy(hFeature);
    }
    free( pCurrWKB );
    VectorParser_destroy(pParser);

    return (PyObject*)pOutMask;
}


/* Our list of functions in this module*/
static PyMethodDef InsidePolyMethods[] = {
    {"insideLayer", insidepoly_insideLayer, METH_VARARGS, 
"read an OGR layer and some points and return a mask array that is True where points are within the polygons:\n"
"call signature: mask = insideLayer(ogrlayer, xvalues, yvalues)\n"
"where:\n"
"  ogrlayer is an instance of ogr.Layer\n"
"  xvalues and yvalues are 1d arrays that contain the points to test"},
    {NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int insidepoly_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int insidepoly_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "insidepoly",
        NULL,
        sizeof(struct InsidePolyState),
        InsidePolyMethods,
        NULL,
        insidepoly_traverse,
        insidepoly_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit_insidepoly(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initinsidepoly(void)
#endif
{
    PyObject *pModule;
    struct InsidePolyState *state;

    /* initialize the numpy stuff */
    import_array();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("inside", InsidePolyMethods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("insidepoly.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
