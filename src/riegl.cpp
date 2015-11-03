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
#include <cmath>
#include <limits>

static const int nGrowBy = 100;

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
    double range;
    double papp;
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
    CREATE_FIELD_DEFN(SRieglPoint, range, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, papp, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, x, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, y, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, z, 'f'),
    {NULL} // Sentinel
};

/* Python object wrapping a scanlib::basic_rconnection */
typedef struct
{
    PyObject_HEAD
    std::shared_ptr<scanlib::basic_rconnection> rc;
    scanlib::decoder_rxpmarker *pDecoder;
    scanlib::buffer *pBuffer;
    Py_ssize_t nTotalPulsesRead;
    bool bFinishedReading;

    uint32_t nscanline;
    uint16_t nscanlineIdx;

    // Riegl often gives us more data than we need
    // for the range of pulses. Stash the extra ones
    // here so we can pick them up next time if needed.
    PylidarVector<SRieglPulse> *pExtraPulses;
    PylidarVector<SRieglPoint> *pExtraPoints;
    Py_ssize_t nStartExtra;

} PyRieglScanFile;

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
#endif

/* destructor - close and delete tc */
static void 
PyRieglScanFile_dealloc(PyRieglScanFile *self)
{
    self->rc->close();
    self->rc.reset();
    delete self->pDecoder;
    delete self->pBuffer;
    delete self->pExtraPoints;
    delete self->pExtraPulses;
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

    try
    {
        self->rc = scanlib::basic_rconnection::create(pszFname);

        // The decoder class scans off distinct packets from the continuous data stream
        // i.e. the rxp format and manages the packets in a buffer.
        self->pDecoder = new scanlib::decoder_rxpmarker(self->rc);

        // The buffer is a structure that holds pointers into the decoder buffer
        // thereby avoiding unnecessary copies of the data.
        self->pBuffer = new scanlib::buffer();
    }
    catch(scanlib::scanlib_exception e)
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_Format(GETSTATE(m)->error, "Error from Riegl lib: %s", e.what());
        return -1;
    }   
    self->nTotalPulsesRead = 0;
    self->bFinishedReading = false;
    self->nscanline = 0;
    self->nscanlineIdx = 0;
    self->nStartExtra = 0;
    self->pExtraPulses = NULL;
    self->pExtraPoints = NULL;
    return 0;
}

class RieglReader : public scanlib::pointcloud
{
public:
    RieglReader(Py_ssize_t nPulses, Py_ssize_t nTotalPulsesRead, Py_ssize_t nPulsesToIgnore,
            uint32_t nscanline, uint16_t nscanlineIdx) : 
        scanlib::pointcloud(false), 
        m_nPulses(nPulses),
        m_nTotalPulsesRead(nTotalPulsesRead),
        m_nPulsesToIgnore(nPulsesToIgnore),
        m_scanline(nscanline),
        m_scanlineIdx(nscanlineIdx),
        m_Pulses(nPulses, nGrowBy),
        m_Points(nPulses, nGrowBy)
    {
    }

    bool done()
    {
        return m_Pulses.getNumElems() >= m_nPulses;
    }

    Py_ssize_t getNumPulsesRead()
    {
        return m_Pulses.getNumElems();
    }

    Py_ssize_t getNumPointsRead()
    {
        return m_Points.getNumElems();
    }

    PyObject *getPulses()
    {
        return m_Pulses.getNumpyArray(RieglPulseFields);
    }

    PyObject *getPoints()
    {
        return m_Points.getNumpyArray(RieglPointFields);
    }

    PylidarVector<SRieglPulse> *splitExtraPulses(npy_intp nUpper)
    {
        return m_Pulses.splitUpper(nUpper);
    }

    PylidarVector<SRieglPoint> *splitExtraPoints(npy_intp nUpper)
    {
        return m_Points.splitUpper(nUpper);
    }

    // add in stuff from previous read split out by splitExtraPoints
    void appendPointsArray(PylidarVector<SRieglPoint> *otherPoints)
    {
        m_Points.appendArray(otherPoints);
    }

    void appendPulsesArray(PylidarVector<SRieglPulse> *otherPulses)
    {
        m_Pulses.appendArray(otherPulses);
        m_nTotalPulsesRead += otherPulses->getNumElems();
    }

    uint32_t getScanline()
    {
        return m_scanline;
    }
    uint16_t getScanlineIdx()
    {
        return m_scanlineIdx;
    }

protected:
    // This call is invoked for every pulse, even if there is no return
    void on_shot()
    {
        m_scanlineIdx++;

        if( m_nPulsesToIgnore > 0 )
        {
            m_nPulsesToIgnore--;
            return;
        }

        SRieglPulse pulse;
        pulse.pulseID = m_Pulses.getNumElems() + m_nTotalPulsesRead;
        pulse.gpsTime = time_sorg * 1e9 + 0.5;

        // Get spherical coordinates. TODO: matrix transform
        double magnitude = std::sqrt(beam_direction[0] * beam_direction[0] + \
                           beam_direction[1] * beam_direction[1] + \
                           beam_direction[2] * beam_direction[2]);
        double shot_zenith = std::acos(beam_direction[2]/magnitude) * 180.0 / pi;
        double shot_azimuth = shot_azimuth = std::atan2(beam_direction[0],beam_direction[1]) * 180.0 / pi;      
        if( beam_direction[0] < 0 )
        {
            shot_azimuth += 360.0;            
        }

        pulse.azimuth = shot_azimuth;
        pulse.zenith = shot_zenith;
        pulse.scanline = m_scanline;
        pulse.scanlineIdx = m_scanlineIdx;
        // do we need these separate?
        pulse.xIdx = m_scanline;
        pulse.yIdx = m_scanlineIdx;
        // TODO: matrix transform
        pulse.xOrigin = beam_origin[0];
        pulse.yOrigin = beam_origin[1];
        pulse.zOrigin = beam_origin[2];

        // point idx - start with 0
        pulse.pointStartIdx = 0;
        pulse.pointCount = 0;

        m_Pulses.push(&pulse);
    }

    // overridden from pointcloud class
    void on_echo_transformed(echo_type echo)
    {
        // we assume that this point will be
        // connected to the last pulse...
        SRieglPulse *pPulse = m_Pulses.getLastElement();
        if(pPulse == NULL)
        {
            throw scanlib::scanlib_exception("Point before Pulse.");
        }
        if(pPulse->pointCount == 0)
        {
            // note: haven't pushed point yet
            pPulse->pointStartIdx = m_Points.getNumElems();
        }
        pPulse->pointCount++;

        SRieglPoint point;

        // the current echo is always indexed by target_count-1.
        scanlib::target& current_target(targets[target_count-1]);

        point.returnId = target_count;
        point.gpsTime = current_target.time * 1e9 + 0.5;
        point.amplitudeReturn = current_target.amplitude;
        point.widthReturn = current_target.deviation;
        point.classification = 1;

        // Get range from optical centre of scanner
        // vertex[i] = beam_origin[i] + echo_range * beam_direction[i]
        double point_range = current_target.echo_range;
        if (point_range <= std::numeric_limits<double>::epsilon()) 
        {
            current_target.vertex[0] = current_target.vertex[1] = current_target.vertex[2] = 0;
            point_range = 0;
        }
        point.range = point_range;

        // Rescale reflectance from dB to papp
        point.papp = std::pow(10.0, current_target.reflectance / 10.0);

        point.x = current_target.vertex[0];
        point.y = current_target.vertex[1];
        point.z = current_target.vertex[2];

        m_Points.push(&point);
    }

    // start of a scan line going in the up direction
    void on_line_start_up(const scanlib::line_start_up<iterator_type>& arg) 
    {
        scanlib::pointcloud::on_line_start_up(arg);
        ++m_scanline;
        m_scanlineIdx = 0;
    }
    
    // start of a scan line going in the down direction
    void on_line_start_dn(const scanlib::line_start_dn<iterator_type>& arg) 
    {
        scanlib::pointcloud::on_line_start_dn(arg);
        ++m_scanline;
        m_scanlineIdx = 0;
    }

private:
    Py_ssize_t m_nPulses;
    Py_ssize_t m_nTotalPulsesRead;
    Py_ssize_t m_nPulsesToIgnore;
    PylidarVector<SRieglPulse> m_Pulses;
    PylidarVector<SRieglPoint> m_Points;
    uint32_t m_scanline;
    uint16_t m_scanlineIdx;
};

void PyRieglScanFile_resetExtra(PyRieglScanFile *self)
{
    delete self->pExtraPulses;
    self->pExtraPulses = NULL;
    delete self->pExtraPoints;
    self->pExtraPoints = NULL;
    self->nStartExtra = 0;
}

static PyObject *PyRieglScanFile_readData(PyRieglScanFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses, nCount;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    Py_ssize_t nPulsesToIgnore = 0; // how many to ignore before doing stuff
    if( nPulseStart < self->nTotalPulsesRead )
    {
        // need to read earlier stuff in the file. 
        // reset to beginning and start looping
        self->rc->seekg(0);
        // ensure buffers are flushed
        delete self->pDecoder;
        delete self->pBuffer;
        self->pDecoder = new scanlib::decoder_rxpmarker(self->rc);
        self->pBuffer = new scanlib::buffer();

        self->nTotalPulsesRead = nPulseStart;
        nPulsesToIgnore = nPulseStart;
        PyRieglScanFile_resetExtra(self);
        self->nscanline = 0;
        self->nscanlineIdx = 0;
    }
    else if( nPulseStart > self->nTotalPulsesRead )
    {
        // requested range is after current location
        nPulsesToIgnore = nPulseStart - self->nTotalPulsesRead;
        self->nTotalPulsesRead = nPulseStart;
        PyRieglScanFile_resetExtra(self);
    }

    // our reader class
    RieglReader reader(nPulses, self->nTotalPulsesRead, nPulsesToIgnore,
                        self->nscanline, self->nscanlineIdx);

    // There is stuff in the 'extra' that leads on
    // add it into the reader's buffer first
    if( ( self->nStartExtra == nPulseStart ) && ( self->pExtraPulses != NULL ) && 
            ( self->pExtraPulses->getNumElems() > 0 ) )
    {
        reader.appendPulsesArray(self->pExtraPulses);
        reader.appendPointsArray(self->pExtraPoints);
        PyRieglScanFile_resetExtra(self);
    }

    // loop through the requested number of pulses
    if( !reader.done() ) // don't bother if 'extra' has given us enough
    {
        try
        {
            for( self->pDecoder->get(*self->pBuffer); 
                !self->pDecoder->eoi() && !reader.done(); self->pDecoder->get(*self->pBuffer) )
            {
                reader.dispatch(self->pBuffer->begin(), self->pBuffer->end());
            }
        }
        catch(scanlib::scanlib_exception e)
        {
            // raise Python exception
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_Format(GETSTATE(m)->error, "Error from Riegl lib: %s", e.what());
            return NULL;
        }
    }

    // OK Riegl often gives us more data than we need because
    // of the way the packet decoding works. Split any extra into
    // the 'extra' arrays.
    if( reader.getNumPulsesRead() > nPulses )
    {
        self->pExtraPulses = reader.splitExtraPulses(nPulses);
        uint32_t nPointIdx = self->pExtraPulses->getFirstElement()->pointStartIdx;
        self->pExtraPoints = reader.splitExtraPoints(nPointIdx);
        // reset all the pointStartIdx fields in the pulses to match
        // the now shorter array of points
        for( npy_intp n = 0; n < self->pExtraPulses->getNumElems(); n++ )
        {
            SRieglPulse *pPulse = self->pExtraPulses->getElem(n);
            if( pPulse->pointCount > 0 )
                pPulse->pointStartIdx -= nPointIdx;
        }

        self->nStartExtra = self->nTotalPulsesRead + nPulses;
    }

    // update how many were actually read
    self->nTotalPulsesRead += reader.getNumPulsesRead();

    // we have finished if we are at the end
    self->bFinishedReading = self->pDecoder->eoi();

    // update our scanline and scanline IDX
    self->nscanline = reader.getScanline();
    self->nscanlineIdx = reader.getScanlineIdx();

    // get pulse array as numpy array
    PyObject *pPulses = reader.getPulses(); 
    // points
    PyObject *pPoints = reader.getPoints();

    // build tuple
    PyObject *pTuple = PyTuple_New(2);
    PyTuple_SetItem(pTuple, 0, pPulses);
    PyTuple_SetItem(pTuple, 1, pPoints);

    return pTuple;
}

/* Table of methods */
static PyMethodDef PyRieglScanFile_methods[] = {
    {"readData", (PyCFunction)PyRieglScanFile_readData, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyObject *PyRieglScanFile_getFinished(PyRieglScanFile *self, void *closure)
{
    if( self->bFinishedReading )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyRieglScanFile_getPulsesRead(PyRieglScanFile *self, void *closure)
{
    return PyLong_FromSsize_t(self->nTotalPulsesRead);
}

/* get/set */
static PyGetSetDef PyRieglScanFile_getseters[] = {
    {"finished", (getter)PyRieglScanFile_getFinished, NULL, "Get Finished reading state", NULL}, 
    {"pulsesRead", (getter)PyRieglScanFile_getPulsesRead, NULL, "Get number of pulses read", NULL},
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
    PyRieglScanFile_getseters,           /* tp_getset */
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
    pModule = Py_InitModule("_riegl", NULL);
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
