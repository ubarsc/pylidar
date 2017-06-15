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

// for CVector
static const int nGrowBy = 1000;
static const int nInitSize = 40000;

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

    npy_uint32 wfm_start_idx;
    npy_uint8 number_of_waveform_samples;
    npy_uint8 number_of_returns;
    npy_uint64 pts_start_idx;
} SPulseWavesPulse;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn PulseWavesPulseFields[] = {
    CREATE_FIELD_DEFN(SPulseWavesPulse, time, 'i'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, offset, 'i'),
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

    CREATE_FIELD_DEFN(SPulseWavesPulse, wfm_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, number_of_waveform_samples, 'u'),    
    CREATE_FIELD_DEFN(SPulseWavesPulse, number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesPulse, pts_start_idx, 'u'),
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
    npy_uint8 channel;
} SPulseWavesWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn PulseWavesWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, number_of_waveform_transmitted_bins, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, transmitted_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, range_to_waveform_start, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, channel, 'u'),
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
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    pylidar::CVector<SPulseWavesPulse> pulses(nPulses, nGrowBy);
    pylidar::CVector<SPulseWavesPoint> points(nPulses, nGrowBy);
    pylidar::CVector<SPulseWavesWaveformInfo> waveformInfos(nPulses, nGrowBy);
    pylidar::CVector<npy_int32> transmitted(nPulses, nGrowBy);
    pylidar::CVector<npy_int32> received(nPulses, nGrowBy);
    SPulseWavesPulse pwPulse;
    SPulseWavesPoint pwPoint;
    SPulseWavesWaveformInfo pwWaveformInfo;

    // seek to the first pulse
    if( !self->pReader->seek(nPulseStart) )
        self->bFinished = true;
    else
    {
        for( Py_ssize_t i = 0; i < nPulses; i++ )
        {
            // read a pulse
            if( !self->pReader->read_pulse() )
            {
                self->bFinished = true;
                break;
            }
    
            // copy it into our data structures
            pwPulse.time = self->pReader->pulse.get_T();
            pwPulse.offset = self->pReader->pulse.offset;
            pwPulse.x_origin = self->pReader->pulse.get_anchor_x();
            pwPulse.y_origin = self->pReader->pulse.get_anchor_y();
            pwPulse.z_origin = self->pReader->pulse.get_anchor_z();
            pwPulse.x_target = self->pReader->pulse.get_target_x();
            pwPulse.y_target = self->pReader->pulse.get_target_y();
            pwPulse.z_target = self->pReader->pulse.get_target_z();
            pwPulse.first_returning_sample = self->pReader->pulse.first_returning_sample;
            pwPulse.last_returning_sample = self->pReader->pulse.last_returning_sample;
            pwPulse.descriptor_index = self->pReader->pulse.descriptor_index;
            pwPulse.intensity = self->pReader->pulse.intensity;
        
            pwPulse.wfm_start_idx = waveformInfos.getNumElems();
            pwPulse.number_of_waveform_samples = 0;
        
            pwPulse.number_of_returns = 1;
            pwPulse.pts_start_idx = points.getNumElems();

            // TODO: is there always just one point per pulse? (no return_number like in LAS)
            // TODO: point x,y,z from anchor or target?
            pwPoint.x = pwPulse.x_origin;
            pwPoint.y = pwPulse.y_origin;
            pwPoint.z = pwPulse.z_origin;
            pwPoint.classification = self->pReader->pulse.classification;

            // now waveforms
            if(self->pReader->read_waves())
            {
                // init these values to 0
                pwWaveformInfo.number_of_waveform_received_bins = 0;
                pwWaveformInfo.received_start_idx = received.getNumElems();
                pwWaveformInfo.number_of_waveform_transmitted_bins = 0;
                pwWaveformInfo.transmitted_start_idx = transmitted.getNumElems();

                for( U16 nSampling = 0; nSampling < self->pReader->waves->get_number_of_samplings(); nSampling++ )
                {
                    WAVESsampling *pSampling = self->pReader->waves->get_sampling(nSampling);
                    for( U16 nSegment = 0; nSegment < pSampling->get_number_of_segments(); nSegment++ )
                    {
                        pSampling->set_active_segment(nSegment);

                        pwWaveformInfo.channel = pSampling->get_channel();
                        pwWaveformInfo.range_to_waveform_start = pSampling->get_duration_from_anchor_for_segment();

                        for( I32 nSample = 0; nSample < pSampling->get_number_of_samples(); nSample++ )
                        {
                            I32 nSampleVal = pSampling->get_sample(nSample);
                            if( pSampling->get_type() == PULSEWAVES_OUTGOING )
                            {
                                transmitted.push(&nSampleVal);
                                pwWaveformInfo.number_of_waveform_transmitted_bins++;
                            }
                            else if( pSampling->get_type() == PULSEWAVES_RETURNING )
                            {
                                received.push(&nSampleVal);
                                pwWaveformInfo.number_of_waveform_received_bins++;
                            }
                            // Not sure if there are other types? Ignore for now
                        }

                        // if the count is 0 then set the index to 0. Not strictly needed...
                        if( pwWaveformInfo.number_of_waveform_transmitted_bins == 0 )
                            pwWaveformInfo.transmitted_start_idx = 0;
                        if( pwWaveformInfo.number_of_waveform_received_bins == 0 )
                            pwWaveformInfo.received_start_idx = 0;

                        waveformInfos.push(&pwWaveformInfo);
                        pwPulse.number_of_waveform_samples++;
                    }
                }
                
            }

            pulses.push(&pwPulse);
            points.push(&pwPoint);
        }
    }
    PyArrayObject *pNumpyPoints = points.getNumpyArray(PulseWavesPointFields);
    PyArrayObject *pNumpyPulses = pulses.getNumpyArray(PulseWavesPulseFields);
    PyArrayObject *pNumpyInfos = waveformInfos.getNumpyArray(PulseWavesWaveformInfoFields);
    PyArrayObject *pNumpyReceived = received.getNumpyArray(NPY_INT32);
    PyArrayObject *pNumpyTransmitted = transmitted.getNumpyArray(NPY_INT32);

    // build tuple
    PyObject *pTuple = PyTuple_Pack(5, pNumpyPulses, pNumpyPoints, pNumpyInfos, pNumpyReceived, pNumpyTransmitted);

    // decref the objects since we have finished with them (PyTuple_Pack increfs)
    Py_DECREF(pNumpyPulses);
    Py_DECREF(pNumpyPoints);
    Py_DECREF(pNumpyInfos);
    Py_DECREF(pNumpyReceived);
    Py_DECREF(pNumpyTransmitted);
    
    return pTuple;
}

/* calculate the length in case they change in future */
#define GET_LENGTH(x) (sizeof(x) / sizeof(x[0]))

static PyObject *PyPulseWavesFileRead_readHeader(PyPulseWavesFileRead *self, PyObject *args)
{
    PyObject *pHeaderDict = PyDict_New();
    PULSEheader *pHeader = &self->pReader->header;

#if PY_MAJOR_VERSION >= 3
    PyObject *pVal = PyUnicode_FromStringAndSize(pHeader->file_signature, 
                GET_LENGTH(pHeader->file_signature));
#else
    PyObject *pVal = PyString_FromStringAndSize(pHeader->file_signature, 
                GET_LENGTH(pHeader->file_signature));
#endif
    PyDict_SetItemString(pHeaderDict, "FILE_SIGNATURE", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->global_parameters);
    PyDict_SetItemString(pHeaderDict, "GLOBAL_PARAMETERS", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->file_source_ID);
    PyDict_SetItemString(pHeaderDict, "FILE_SOURCE_ID", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->project_ID_GUID_data_1);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_1", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->project_ID_GUID_data_2);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_2", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->project_ID_GUID_data_3);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_3", pVal);
    Py_DECREF(pVal);

#if PY_MAJOR_VERSION >= 3
    pVal = PyUnicode_FromStringAndSize(pHeader->system_identifier, 
                GET_LENGTH(pHeader->system_identifier));
#else
    pVal = PyString_FromStringAndSize(pHeader->system_identifier, 
                GET_LENGTH(pHeader->system_identifier));
#endif
    PyDict_SetItemString(pHeaderDict, "SYSTEM_IDENTIFIER", pVal);
    Py_DECREF(pVal);

#if PY_MAJOR_VERSION >= 3
    pVal = PyUnicode_FromStringAndSize(pHeader->generating_software, 
                GET_LENGTH(pHeader->generating_software));
#else
    pVal = PyString_FromStringAndSize(pHeader->generating_software, 
                GET_LENGTH(pHeader->generating_software));
#endif
    PyDict_SetItemString(pHeaderDict, "GENERATING_SOFTWARE", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->file_creation_day);
    PyDict_SetItemString(pHeaderDict, "FILE_CREATION_DAY", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->file_creation_year);
    PyDict_SetItemString(pHeaderDict, "FILE_CREATION_YEAR", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->header_size);
    PyDict_SetItemString(pHeaderDict, "HEADER_SIZE", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromLongLong(pHeader->offset_to_pulse_data);
    PyDict_SetItemString(pHeaderDict, "OFFSET_TO_PULSE_DATA", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromLongLong(pHeader->number_of_pulses);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_PULSES", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->pulse_format);
    PyDict_SetItemString(pHeaderDict, "PULSE_FORMAT", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->pulse_attributes);
    PyDict_SetItemString(pHeaderDict, "PULSE_ATTRIBUTES", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->pulse_size);
    PyDict_SetItemString(pHeaderDict, "PULSE_SIZE", pVal);
    Py_DECREF(pVal);
    
    pVal = PyLong_FromUnsignedLong(pHeader->pulse_compression);
    PyDict_SetItemString(pHeaderDict, "PULSE_COMPRESSION", pVal);
    Py_DECREF(pVal);
    
    pVal = PyLong_FromUnsignedLong(pHeader->number_of_variable_length_records);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_VARIABLE_LENGTH_RECORDS", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromLong(pHeader->number_of_appended_variable_length_records);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_APPENDED_VARIABLE_LENGTH_RECORDS", pVal);
    Py_DECREF(pVal);
    
    pVal = PyFloat_FromDouble(pHeader->max_x);
    PyDict_SetItemString(pHeaderDict, "X_MAX", pVal);
    Py_DECREF(pVal);

    pVal = PyFloat_FromDouble(pHeader->min_x);
    PyDict_SetItemString(pHeaderDict, "X_MIN", pVal);
    Py_DECREF(pVal);

    pVal = PyFloat_FromDouble(pHeader->max_y);
    PyDict_SetItemString(pHeaderDict, "Y_MAX", pVal);
    Py_DECREF(pVal);

    pVal = PyFloat_FromDouble(pHeader->min_y);
    PyDict_SetItemString(pHeaderDict, "Y_MIN", pVal);
    Py_DECREF(pVal);

    pVal = PyFloat_FromDouble(pHeader->max_z);
    PyDict_SetItemString(pHeaderDict, "Z_MAX", pVal);
    Py_DECREF(pVal);

    pVal = PyFloat_FromDouble(pHeader->min_z);
    PyDict_SetItemString(pHeaderDict, "Z_MIN", pVal);
    Py_DECREF(pVal);

    return pHeaderDict;
}

/* Table of methods */
static PyMethodDef PyPulseWavesFileRead_methods[] = {
    {"readData", (PyCFunction)PyPulseWavesFileRead_readData, METH_VARARGS, NULL},
    {"readHeader", (PyCFunction)PyPulseWavesFileRead_readHeader, METH_NOARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyObject *PyPulseWavesFileRead_getFinished(PyPulseWavesFileRead* self, void *closure)
{
    if( self->bFinished )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyPulseWavesFileRead_getNumPulses(PyPulseWavesFileRead* self, void *closure)
{
    return PyLong_FromLongLong(self->pReader->npulses);
}

static PyGetSetDef PyPulseWavesFileRead_getseters[] = {
    {(char*)"finished", (getter)PyPulseWavesFileRead_getFinished, NULL, (char*)"Get Finished reading state", NULL},
    {(char*)"numPulses", (getter)PyPulseWavesFileRead_getNumPulses, NULL, (char*)"Get number of pulses in file", NULL},
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
    PyPulseWavesFileRead_getseters,           /* tp_getset */
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
    PyModule_AddObject(pModule, "FileRead", (PyObject *)&PyPulseWavesFileReadType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}

