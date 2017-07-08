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
#include "pylfieldinfomap.h"

#include "pulsereader.hpp"
#include "pulsewriter.hpp"

#define POINT_FROM_ANCHOR 0
#define POINT_FROM_TARGET 1

// for CVector
static const int nGrowBy = 10000;
static const int nInitSize = 256*256;

/* An exception object for this module */
/* created in the init function */
struct PulseWavesState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct PulseWavesState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
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
    float receive_wave_gain;
    float receive_wave_offset;
    float trans_wave_gain;
    float trans_wave_offset;
} SPulseWavesWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn PulseWavesWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, number_of_waveform_transmitted_bins, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, transmitted_start_idx, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, range_to_waveform_start, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, channel, 'u'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, receive_wave_gain, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, receive_wave_offset, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, trans_wave_gain, 'f'),
    CREATE_FIELD_DEFN(SPulseWavesWaveformInfo, trans_wave_offset, 'f'),
    {NULL} // Sentinel
};

// Python object wrapping a PULSEreader
typedef struct {
    PyObject_HEAD
    PULSEreader *pReader;
    bool bFinished;
    int nPointFrom;
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
int nPointFrom = POINT_FROM_ANCHOR;

    if( !PyArg_ParseTuple(args, "s|i", &pszFname, &nPointFrom) )
    {
        return -1;
    }

    self->bFinished = false;
    self->nPointFrom = nPointFrom;

    PULSEreadOpener pulsereadopener;
    pulsereadopener.set_file_name(pszFname);

    self->pReader = pulsereadopener.open();
    if( self->pReader == NULL )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Unable to open pulsewaves file");
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
    // we don't support these fields but they are required
    // by SPDV4 so set them so they do nothing.
    pwWaveformInfo.receive_wave_gain = 1.0;
    pwWaveformInfo.receive_wave_offset = 0.0;
    pwWaveformInfo.trans_wave_gain = 1.0;
    pwWaveformInfo.trans_wave_offset = 0.0;


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

            self->pReader->pulse.compute_anchor_and_target(); // important
    
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
            if( self->nPointFrom == POINT_FROM_ANCHOR )
            {
                pwPoint.x = self->pReader->pulse.get_anchor_x();
                pwPoint.y = self->pReader->pulse.get_anchor_y();
                pwPoint.z = self->pReader->pulse.get_anchor_z();
            }
            else
            {
                pwPoint.x = self->pReader->pulse.get_target_x();
                pwPoint.y = self->pReader->pulse.get_target_y();
                pwPoint.z = self->pReader->pulse.get_target_z();
            }
            pwPoint.classification = self->pReader->pulse.classification;

            // now waveforms - we have to do this every time whether or not
            // waveforms are required right now since this is the only time to get them.
            if(self->pReader->read_waves())
            {
                for( U16 nSampling = 0; nSampling < self->pReader->waves->get_number_of_samplings(); nSampling++ )
                {
                    WAVESsampling *pSampling = self->pReader->waves->get_sampling(nSampling);
                    for( U16 nSegment = 0; nSegment < pSampling->get_number_of_segments(); nSegment++ )
                    {
                        pSampling->set_active_segment(nSegment);

                        pwWaveformInfo.channel = pSampling->get_channel();
                        pwWaveformInfo.range_to_waveform_start = pSampling->get_duration_from_anchor_for_segment();
                        // init these values
                        pwWaveformInfo.number_of_waveform_received_bins = 0;
                        pwWaveformInfo.received_start_idx = received.getNumElems();
                        pwWaveformInfo.number_of_waveform_transmitted_bins = 0;
                        pwWaveformInfo.transmitted_start_idx = transmitted.getNumElems();

                        for( I32 nSample = 0; nSample < pSampling->get_number_of_samples(); nSample++ )
                        {
                            npy_int32 nSampleVal = pSampling->get_sample(nSample);
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

                        if( ( pwWaveformInfo.number_of_waveform_transmitted_bins > 0 ) ||
                            ( pwWaveformInfo.number_of_waveform_received_bins > 0 ) )
                        {
                            // only do this if we did actually get a waveform we can use
                            // - see note about other types from pSampling->get_type() above
                            waveformInfos.push(&pwWaveformInfo);
                            pwPulse.number_of_waveform_samples++;
                        }
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

    pylidar::CVector<U8> project_ID_GUID_data_4Vector(pHeader->project_ID_GUID_data_4, 
                            sizeof(pHeader->project_ID_GUID_data_4));    
    pVal = (PyObject*)project_ID_GUID_data_4Vector.getNumpyArray(NPY_UINT8);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_4", pVal);
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

/* Python object wrapping a PULSEwriter */
typedef struct {
    PyObject_HEAD
    // set by _init so we can create PULSEwriteOpener
    // when writing the first block
    char *pszFilename;
    // created when writing first block
    PULSEwriter *pWriter;
    PULSEheader *pHeader;
} PyPulseWavesFileWrite;

/* destructor - close and delete */
static void 
PyPulseWavesFileWrite_dealloc(PyPulseWavesFileWrite *self)
{
    if(self->pszFilename != NULL)
    {
        free(self->pszFilename);
    }
    if(self->pWriter != NULL)
    {
        self->pWriter->update_header(self->pHeader, TRUE);
        self->pWriter->close();
        delete self->pWriter;
    }
    if(self->pHeader != NULL)
    {
        delete self->pHeader;
    }
}

/* init method - open file */
static int 
PyPulseWavesFileWrite_init(PyPulseWavesFileWrite *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname ) )
    {
        return -1;
    }

    // copy filename so we can open later
    self->pszFilename = strdup(pszFname);
    self->pWriter = NULL;
    self->pHeader = NULL;

    return 0;
}

// copies recognised fields from pHeaderDict into pHeader
void setHeaderFromDictionary(PyObject *pHeaderDict, PULSEheader *pHeader)
{
    PyObject *pVal;
    // TODO: be better at checking types
    // TODO: error if key not recognised? maybe copying (PyDict_Copy) then
    // deleting recognised keys - if stuff left then error etc.

    pVal = PyDict_GetItemString(pHeaderDict, "FILE_SIGNATURE");
    if( pVal != NULL )
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *bytesKey = PyUnicode_AsEncodedString(pVal, NULL, NULL);
        char *pszSignature = PyBytes_AsString(bytesKey);
#else
        char *pszSignature = PyString_AsString(pVal);
#endif
        strncpy(pHeader->file_signature, pszSignature, GET_LENGTH(pHeader->file_signature));
#if PY_MAJOR_VERSION >= 3
        Py_DECREF(bytesKey);
#endif
    }

    pVal = PyDict_GetItemString(pHeaderDict, "GLOBAL_PARAMETERS");
    if( pVal != NULL )
        pHeader->global_parameters = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "FILE_SOURCE_ID");
    if( pVal != NULL )
        pHeader->file_source_ID = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_1");
    if( pVal != NULL )
        pHeader->project_ID_GUID_data_1 = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_2");
    if( pVal != NULL )
        pHeader->project_ID_GUID_data_2 = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_3");
    if( pVal != NULL )
        pHeader->project_ID_GUID_data_3 = PyLong_AsLong(pVal);


    pVal = PyDict_GetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_4");
    if( pVal != NULL )
    {    
        PyObject *pArray = PyArray_FROM_OT(pVal, NPY_UINT8);
        // TODO: check 1d?
        for( npy_intp i = 0; (i < PyArray_DIM((PyArrayObject*)pArray, 0)) &&
                            (i < (npy_intp)GET_LENGTH(pHeader->project_ID_GUID_data_4)); i++ )
        {
            pHeader->project_ID_GUID_data_4[i] = *((U8*)PyArray_GETPTR1((PyArrayObject*)pArray, i));
        }
        Py_DECREF(pArray);
    }

    pVal = PyDict_GetItemString(pHeaderDict, "SYSTEM_IDENTIFIER");
    if( pVal != NULL )
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *bytesKey = PyUnicode_AsEncodedString(pVal, NULL, NULL);
        char *pszIdent = PyBytes_AsString(bytesKey);
#else
        char *pszIdent = PyString_AsString(pVal);
#endif
        strncpy(pHeader->system_identifier, pszIdent, GET_LENGTH(pHeader->system_identifier));
#if PY_MAJOR_VERSION >= 3
        Py_DECREF(bytesKey);
#endif
    }

    pVal = PyDict_GetItemString(pHeaderDict, "GENERATING_SOFTWARE");
    if( pVal != NULL )
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *bytesKey = PyUnicode_AsEncodedString(pVal, NULL, NULL);
        char *pszSW = PyBytes_AsString(bytesKey);
#else
        char *pszSW = PyString_AsString(pVal);
#endif
        strncpy(pHeader->generating_software, pszSW, GET_LENGTH(pHeader->generating_software));
#if PY_MAJOR_VERSION >= 3
        Py_DECREF(bytesKey);
#endif
    }

    pVal = PyDict_GetItemString(pHeaderDict, "FILE_CREATION_DAY");
    if( pVal != NULL )
        pHeader->file_creation_day = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "FILE_CREATION_YEAR");
    if( pVal != NULL )
        pHeader->file_creation_year = PyLong_AsLong(pVal);

    // should this be set at all?
    pVal = PyDict_GetItemString(pHeaderDict, "HEADER_SIZE");
    if( pVal != NULL )
        pHeader->header_size = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "OFFSET_TO_PULSE_DATA");
    if( pVal != NULL )
        pHeader->offset_to_pulse_data = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "NUMBER_OF_PULSES");
    if( pVal != NULL )
        pHeader->number_of_pulses = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PULSE_FORMAT");
    if( pVal != NULL )
        pHeader->pulse_format = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PULSE_ATTRIBUTES");
    if( pVal != NULL )
        pHeader->pulse_attributes = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PULSE_SIZE");
    if( pVal != NULL )
        pHeader->pulse_size = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "PULSE_COMPRESSION");
    if( pVal != NULL )
        pHeader->pulse_size = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "NUMBER_OF_VARIABLE_LENGTH_RECORDS");
    if( pVal != NULL )
        pHeader->number_of_variable_length_records = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "X_MAX");
    if( pVal != NULL )
        pHeader->max_x = PyFloat_AsDouble(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "X_MIN");
    if( pVal != NULL )
        pHeader->min_x = PyFloat_AsDouble(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "Y_MAX");
    if( pVal != NULL )
        pHeader->max_y = PyFloat_AsDouble(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "Y_MIN");
    if( pVal != NULL )
        pHeader->min_y = PyFloat_AsDouble(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "Z_MAX");
    if( pVal != NULL )
        pHeader->max_z = PyFloat_AsDouble(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "Z_MIN");
    if( pVal != NULL )
        pHeader->min_z = PyFloat_AsDouble(pVal);
}

// an implementation of PulseWaves WAVESsampling class
class PyLidarWAVESsampling : public WAVESsampling
{
public:
    PyLidarWAVESsampling(U8 inchannel, F32 inrange_to_waveform_start, U8 intype,
            U32 innumber_of_samples, U8 *insamples) :
        WAVESsampling()
    {
        WAVESsampling::clean();
        bits_per_sample = 16; // not sure why this can't be 32...
        channel = inchannel;
        type = intype;
        number_of_samples = innumber_of_samples;
        samples = insamples;
        number_of_segments = 1; // I think this is right, we have lots of samples each with 1 segment
        range_to_waveform_start = inrange_to_waveform_start;
    }
    ~PyLidarWAVESsampling()
    {
        free(samples);
    }

    BOOL set_active_segment(U16 segment_idx) { if (segment_idx == 0) return TRUE; return FALSE; };
    U16 get_active_segment() const { return 0; };
    F32 get_duration_from_anchor_for_segment() const { return range_to_waveform_start; };
    BOOL set_number_of_segments(I32 number_of_segments) { if (number_of_segments != 1) return FALSE; return TRUE; };
    BOOL set_duration_from_anchor_for_segment(F32 duration) { quantized_duration = I32_QUANTIZE((duration - offset_for_duration_from_anchor)/scale_for_duration_from_anchor); return TRUE; };
    BOOL set_quantized_duration_from_anchor_for_segment(I32 quantized_duration) { this->quantized_duration = quantized_duration; range_to_waveform_start = scale_for_duration_from_anchor*quantized_duration + offset_for_duration_from_anchor; return TRUE; };
    I32 get_quantized_duration_from_anchor_for_segment() const { return quantized_duration; };
    BOOL set_number_of_samples_for_segment(I32 num_samples) { if ((I32)this->number_of_samples != num_samples) return FALSE; return TRUE; };
    I32 get_number_of_samples_for_segment() const { return number_of_samples; };
    U32 size() const { return number_of_samples * 2; }; // note: 16bit
    U8* get_samples() const { return samples; };
    I32 get_sample(I32 sample_idx) { if (sample_idx < (I32)number_of_samples) { sample = ((U16*)samples)[sample_idx]; return sample; } return 0; };
    BOOL get_sample_xyz(const PULSEpulse* pulse, I32 sample_idx)
    {
        xyz[0] = pulse->get_anchor_x() + pulse->get_dir_x()*(range_to_waveform_start + sample_idx);
        xyz[1] = pulse->get_anchor_y() + pulse->get_dir_y()*(range_to_waveform_start + sample_idx);
        xyz[2] = pulse->get_anchor_z() + pulse->get_dir_z()*(range_to_waveform_start + sample_idx);
        return TRUE;
    }

protected:
    F32 range_to_waveform_start;
    I32 quantized_duration;
    U8 *samples;
};

U16* getWaveformAsArray(PyObject *pWaveform, U16 nSample, npy_intp nPulseIdx, npy_int64 nBins)
{
    // takes a 3d waveform array (transmitted or received)
    // and returns a new array of 16bit waveform numbers
    // caller to free
    U16 *pVals = (U16*)malloc(nBins * sizeof(U16));
    for( npy_int64 nIndex = 0; nIndex < nBins; nIndex++ )
    {
        pVals[nIndex] = *((U16*)PyArray_GETPTR3((PyArrayObject*)pWaveform, nIndex, nSample, nPulseIdx));
    }

    return pVals;
}

// An implementation of PulseWaves WAVESwaves class that gets the
// info out of the waveforminfo array
class PyLidarWAVESwaves : public WAVESwaves
{
public:
    PyLidarWAVESwaves(U16 nSamplings, PyObject *pWaveformInfos,
            pylidar::CFieldInfoMap &waveInfoMap, npy_intp nPulseIdx,
            PyObject *pReceived, PyObject *pTransmitted) :
        WAVESwaves()
    {
        number_of_extra_bytes = 0;
        extra_bytes = NULL;
        // this is incremented each time we get a new one
        number_of_samplings = 0; 
        // worst case scenario there will be a transmitted and received for each waveforminfo
        // saves trying to realloc etc
        samplings = new WAVESsampling*[nSamplings * 2]; 
        for( U16 n = 0; n < nSamplings; n++ )
        {
            void *pInfoRow = PyArray_GETPTR2((PyArrayObject*)pWaveformInfos, n, nPulseIdx);

            npy_int64 channel = waveInfoMap.getIntValue("CHANNEL", pInfoRow);
            double range_to_waveform_start = waveInfoMap.getDoubleValue("RANGE_TO_WAVEFORM_START", pInfoRow);
            npy_int64 number_of_waveform_received_bins = waveInfoMap.getIntValue("NUMBER_OF_WAVEFORM_RECEIVED_BINS", pInfoRow);
            npy_int64 number_of_waveform_transmitted_bins = waveInfoMap.getIntValue("NUMBER_OF_WAVEFORM_TRANSMITTED_BINS", pInfoRow);

            // WAVESwaves destructor deletes samplings
            // PyLidarWAVESsampling frees pData
            if( number_of_waveform_received_bins > 0 )
            {
                U8 *pData = (U8*)getWaveformAsArray(pReceived, n, nPulseIdx, 
                                    number_of_waveform_received_bins);

                samplings[number_of_samplings++] = new PyLidarWAVESsampling(channel, 
                            range_to_waveform_start, PULSEWAVES_RETURNING, 
                            number_of_waveform_received_bins, pData);
            }

            if( number_of_waveform_transmitted_bins > 0 )
            {
                U8 *pData = (U8*)getWaveformAsArray(pTransmitted, n, nPulseIdx, 
                                    number_of_waveform_transmitted_bins);

                samplings[number_of_samplings++] = new PyLidarWAVESsampling(channel, 
                            range_to_waveform_start, PULSEWAVES_OUTGOING,
                            number_of_waveform_transmitted_bins, pData);
            }
        }
    }

};

static PyObject *PyPulseWavesFileWrite_writeData(PyPulseWavesFileWrite *self, PyObject *args)
{
    PyObject *pHeader, *pPulses, *pPoints, *pWaveformInfos, *pReceived, *pTransmitted;
    if( !PyArg_ParseTuple(args, "OOOOOO:writeData", &pHeader, &pPulses, &pPoints, 
                    &pWaveformInfos, &pReceived, &pTransmitted ) )
        return NULL;

    if( pHeader != Py_None )
    {
        if( !PyDict_Check(pHeader) )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "First parameter to writeData must be header dictionary");
            return NULL;
        }
    }

    bool bArraysOk = true;
    const char *pszMessage = "";
    if( (pPulses == Py_None) || (pPoints == Py_None) )
    {
        bArraysOk = false;
        pszMessage = "Both points and pulses must be set for writing";
    }
    if( bArraysOk && (!PyArray_Check(pPulses) || !PyArray_Check(pPoints)))
    {
        bArraysOk = false;
        pszMessage = "Both points and pulses must be numpy arrays";
    }
    bool bHaveWaveformInfos = (pWaveformInfos != Py_None);
    bool bHaveReceived = (pReceived != Py_None);
    bool bHaveTransmitted = (pTransmitted != Py_None);
    if( bArraysOk && bHaveWaveformInfos && !bHaveReceived && !bHaveTransmitted )
    {
        bArraysOk = false;
        pszMessage = "if a waveform info is set, then either must have transmitted or received";
    }
    if( bArraysOk && !bHaveWaveformInfos && (bHaveReceived || bHaveTransmitted) )
    {
        bArraysOk = false;
        pszMessage = "if a waveform info not set, then don't pass received or transmitted";
    }
    if( bArraysOk && bHaveWaveformInfos && !PyArray_Check(pWaveformInfos) )
    {
        bArraysOk = false;
        pszMessage = "Waveform info must be a numpy array";
    }
    if( bArraysOk && bHaveReceived && !PyArray_Check(pReceived) )
    {
        bArraysOk = false;
        pszMessage = "received must be a numpy array";
    }
    if( bArraysOk && bHaveTransmitted && !PyArray_Check(pTransmitted) )
    {
        bArraysOk = false;
        pszMessage = "transmitted must be a numpy array";
    }
    if( bArraysOk && ((PyArray_NDIM((PyArrayObject*)pPulses) != 1) || (PyArray_NDIM((PyArrayObject*)pPoints) != 2) || 
            (bHaveWaveformInfos && (PyArray_NDIM((PyArrayObject*)pWaveformInfos) != 2)) || 
            (bHaveReceived && (PyArray_NDIM((PyArrayObject*)pReceived) != 3)) ||
            (bHaveTransmitted && (PyArray_NDIM((PyArrayObject*)pTransmitted) != 3)) ) )
    {
        bArraysOk = false;
        pszMessage = "pulses must be 1d, points and waveforminfo 2d, received and transmitted 3d";
    }
    if( bArraysOk && bHaveReceived && (PyArray_TYPE((PyArrayObject*)pReceived) != NPY_UINT16))
    {
        // should be set to uint16 by pulsewaves.py
        bArraysOk = false;
        pszMessage = "received must be 16bit";
    }
    if( bArraysOk && bHaveTransmitted && (PyArray_TYPE((PyArrayObject*)pTransmitted) != NPY_UINT16))
    {
        // should be set to uint16 by pulsewaves.py
        bArraysOk = false;
        pszMessage = "received must be 16bit";
    }

    if( !bArraysOk )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, pszMessage);
        return NULL;
    }

    pylidar::CFieldInfoMap pulseMap((PyArrayObject*)pPulses);
    pylidar::CFieldInfoMap pointMap((PyArrayObject*)pPoints);
    pylidar::CFieldInfoMap waveInfoMap((PyArrayObject*)pWaveformInfos);

    if( self->pWriter == NULL )
    {
        // create header for opening file
        self->pHeader = new PULSEheader;
        // populate header from pHeader dictionary
        if( pHeader != Py_None )
        {
            setHeaderFromDictionary(pHeader, self->pHeader);
        }

        // create writer
        PULSEwriteOpener writeOpener;
        writeOpener.set_file_name(self->pszFilename);

        self->pWriter = writeOpener.open(self->pHeader);
        if( self->pWriter == NULL )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "Unable to open pulsewaves file");
            return NULL;
        }

        if( bHaveWaveformInfos )
        {
            if( !writeOpener.open_waves(self->pWriter) )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "Unable to open waves file");
                return NULL;
            }
        }

    }
    PULSEpulse pulse;
    pulse.init(self->pHeader);

    // now write all the pulses
    for( npy_intp nPulseIdx = 0; nPulseIdx < PyArray_DIM((PyArrayObject*)pPulses, 0); nPulseIdx++)
    {
        void *pPulseRow = PyArray_GETPTR1((PyArrayObject*)pPulses, nPulseIdx);

        pulse.set_T(pulseMap.getDoubleValue("TIME", pPulseRow));
        pulse.set_anchor_x(pulseMap.getDoubleValue("X_ORIGIN", pPulseRow));
        pulse.set_anchor_y(pulseMap.getDoubleValue("Y_ORIGIN", pPulseRow));
        pulse.set_anchor_z(pulseMap.getDoubleValue("Z_ORIGIN", pPulseRow));
        pulse.set_target_x(pulseMap.getDoubleValue("X_TARGET", pPulseRow));
        pulse.set_target_y(pulseMap.getDoubleValue("Y_TARGET", pPulseRow));
        pulse.set_target_z(pulseMap.getDoubleValue("Z_TARGET", pPulseRow));
        pulse.first_returning_sample = pulseMap.getIntValue("FIRST_RETURNING_SAMPLE", pPulseRow);
        pulse.last_returning_sample = pulseMap.getIntValue("FIRST_RETURNING_SAMPLE", pPulseRow);
        pulse.descriptor_index = pulseMap.getIntValue("DESCRIPTOR_INDEX", pPulseRow);
        pulse.intensity = pulseMap.getIntValue("INTENSITY", pPulseRow);

        // classification from point
        void *pPointRow = PyArray_GETPTR2((PyArrayObject*)pPoints, 0, nPulseIdx);
        pulse.classification = pointMap.getIntValue("CLASSIFICATION", pPointRow);

        if( !self->pWriter->write_pulse(&pulse) )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "Failed to write pulse");
            return NULL;
        }

        self->pWriter->update_inventory(&pulse);

        // now waves
        if( bHaveWaveformInfos ) 
        {
            npy_int64 nNumWaveSamples = pulseMap.getIntValue("NUMBER_OF_WAVEFORM_SAMPLES", pPulseRow);
            PyLidarWAVESwaves waveswaves(nNumWaveSamples, pWaveformInfos,
                    waveInfoMap, nPulseIdx, pReceived, pTransmitted);

            if( !self->pWriter->write_waves(&waveswaves) )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "Failed to write wave");
                return NULL;
            }
        }

    }

    Py_RETURN_NONE;
}

/* Table of methods */
static PyMethodDef PyPulseWavesFileWrite_methods[] = {
    {"writeData", (PyCFunction)PyPulseWavesFileWrite_writeData, METH_VARARGS, NULL}, 
    {NULL}  /* Sentinel */
};

static PyTypeObject PyPulseWavesFileWriteType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_pulsewaves.FileWrite",         /*tp_name*/
    sizeof(PyPulseWavesFileWrite),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyPulseWavesFileWrite_dealloc, /*tp_dealloc*/
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
    "PulseWaves File Writer object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyPulseWavesFileWrite_methods,             /* tp_methods */
    0,             /* tp_members */
    0,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyPulseWavesFileWrite_init,      /* tp_init */
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
    pModule = Py_InitModule("_pulsewaves", NULL);
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

    /* PulseWaves file write type */
    PyPulseWavesFileWriteType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyPulseWavesFileWriteType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyPulseWavesFileWriteType);
    PyModule_AddObject(pModule, "FileWrite", (PyObject *)&PyPulseWavesFileWriteType);

    // module constants
    PyModule_AddIntConstant(pModule, "POINT_FROM_ANCHOR", POINT_FROM_ANCHOR);
    PyModule_AddIntConstant(pModule, "POINT_FROM_TARGET", POINT_FROM_TARGET);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}

