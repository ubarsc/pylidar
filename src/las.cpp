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

#define _USE_MATH_DEFINES // for Windows
#include <cmath>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "pylvector.h"
#include "pylfieldinfomap.h"

#include "lasreader.hpp"
#include "laswriter.hpp"

// for CVector
static const int nGrowBy = 10000;
static const int nInitSize = 256*256;
// for creating pulses
static const long FIRST_RETURN = 0;
static const long LAST_RETURN = 1;

/* An exception object for this module */
/* created in the init function */
struct LasState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct LasState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
static struct LasState _state;
#endif

/* Structure for LAS pulses */
typedef struct {
    npy_int8 scan_angle_rank;
    npy_int16 scan_angle; // different to scan_angle_rank for extended (las 1.4)
    npy_uint32 pts_start_idx;
    npy_uint8 number_of_returns;
    npy_uint8 orig_number_of_returns; // original in file. != number_of_returns when BUILD_PULSES=False
    double gps_time;
    npy_uint8 scan_direction_flag;
    npy_uint8 edge_of_flight_line;
    npy_uint8 scanner_channel;
    double x_idx;
    double y_idx;
    double x_origin; // the following only set when we have waveforms, or multiple returns
    double y_origin;
    double z_origin;
    double azimuth;
    double zenith;
    npy_uint8 number_of_waveform_samples;
    npy_uint64 wfm_start_idx;
} SLasPulse;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LasPulseFields[] = {
    CREATE_FIELD_DEFN(SLasPulse, scan_angle_rank, 'i'),
    CREATE_FIELD_DEFN(SLasPulse, scan_angle, 'i'),
    CREATE_FIELD_DEFN(SLasPulse, number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, pts_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, orig_number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, gps_time, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, scan_direction_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, edge_of_flight_line, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, scanner_channel, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, x_idx, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, y_idx, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, x_origin, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, y_origin, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, z_origin, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, azimuth, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, zenith, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, number_of_waveform_samples, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, wfm_start_idx, 'u'),
    {NULL} // Sentinel
};

/* Structure for LAS points */
typedef struct {
    double x;
    double y;
    double z;
    npy_uint16 intensity;
    npy_uint8 return_number;
    npy_uint8 classification;
    npy_uint8 synthetic_flag;
    npy_uint8 keypoint_flag;
    npy_uint8 withheld_flag;
    npy_uint8 user_data;
    npy_uint16 point_source_ID;
    npy_uint32 deleted_flag;
    npy_uint8 extended_point_type; // appropriate extended field copied in if set for above fields
    npy_uint16 red;
    npy_uint16 green;
    npy_uint16 blue;
    npy_uint16 nir;
} SLasPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LasPointFields[] = {
    CREATE_FIELD_DEFN(SLasPoint, x, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, y, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, z, 'f'),
    CREATE_FIELD_DEFN(SLasPoint, intensity, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, return_number, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, classification, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, synthetic_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, keypoint_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, withheld_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, user_data, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, point_source_ID, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, deleted_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, extended_point_type, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, red, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, green, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, blue, 'u'),
    CREATE_FIELD_DEFN(SLasPoint, nir, 'u'),
    {NULL} // Sentinel
};

/* Structure for waveform Info */
typedef struct {
    npy_uint32 number_of_waveform_received_bins;
    npy_uint64 received_start_idx;
    double      range_to_waveform_start;
    double      receive_wave_gain;
    double      receive_wave_offset;
} SLasWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn LasWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SLasWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SLasWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLasWaveformInfo, range_to_waveform_start, 'f'),
    CREATE_FIELD_DEFN(SLasWaveformInfo, receive_wave_gain, 'f'),
    CREATE_FIELD_DEFN(SLasWaveformInfo, receive_wave_offset, 'f'),
    {NULL} // Sentinel
};

/* Python object wrapping a LASreader */
typedef struct {
    PyObject_HEAD
    LASreader *pReader;
    LASwaveform13reader *pWaveformReader;
    bool bBuildPulses;
    bool bFinished;
    Py_ssize_t nPulsesRead;
    double fBinSize;
    long nPulseIndex; // FIRST_RETURN or LAST_RETURNs
    SpylidarFieldDefn *pLasPointFieldsWithExt; // != NULL and use instead of LasPointFields when extended fields defined
    std::map<std::string, int> *pExtraPointNativeTypes; // if pLasPointFieldsWithExt != typenums of the extra fields
} PyLasFileRead;

static const char *SupportedDriverOptionsRead[] = {"BUILD_PULSES", "BIN_SIZE", "PULSE_INDEX", NULL};
static PyObject *las_getReadSupportedOptions(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(SupportedDriverOptionsRead);
}

static const char *SupportedDriverOptionsWrite[] = {"FORMAT_VERSION", "RECORD_LENGTH", "WAVEFORM_DESCR", NULL};
static PyObject *las_getWriteSupportedOptions(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(SupportedDriverOptionsWrite);
}

#define N_WAVEFORM_BINS "NUMBER_OF_WAVEFORM_RECEIVED_BINS"
#define RECEIVE_WAVE_GAIN "RECEIVE_WAVE_GAIN"
#define RECEIVE_WAVE_OFFSET "RECEIVE_WAVE_OFFSET"
static const char *ExpectedWaveformFieldsForDescr[] = {N_WAVEFORM_BINS, RECEIVE_WAVE_GAIN, RECEIVE_WAVE_OFFSET, NULL};
static PyObject *las_getExpectedWaveformFieldsForDescr(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(ExpectedWaveformFieldsForDescr);
}

// module methods
static PyMethodDef module_methods[] = {
    {"getReadSupportedOptions", (PyCFunction)las_getReadSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options for reading"},
    {"getWriteSupportedOptions", (PyCFunction)las_getWriteSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options for writing"},
    {"getExpectedWaveformFieldsForDescr", (PyCFunction)las_getExpectedWaveformFieldsForDescr, METH_NOARGS,
        "Get a tuple of the expected fields the waveform table should have for building unique descriptor table."},
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
PyLasFileRead_dealloc(PyLasFileRead *self)
{
    if(self->pReader != NULL)
    {
        self->pReader->close();
        delete self->pReader;
    }
    if(self->pWaveformReader != NULL)
    {
        delete self->pWaveformReader;
    }
    if(self->pLasPointFieldsWithExt != NULL)
    {
        free(self->pLasPointFieldsWithExt);
    }
    if(self->pExtraPointNativeTypes != NULL)
    {
        delete self->pExtraPointNativeTypes;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* init method - open file */
static int 
PyLasFileRead_init(PyLasFileRead *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;
PyObject *pOptionDict;

    if( !PyArg_ParseTuple(args, "sO", &pszFname, &pOptionDict ) )
    {
        return -1;
    }

    if( !PyDict_Check(pOptionDict) )
    {
        PyErr_SetString(GETSTATE_FC->error, "Last parameter to init function must be a dictionary");
        return -1;
    }

    self->bFinished = false;
    self->bBuildPulses = true;
    self->nPulsesRead = 0;
    self->fBinSize = 0;
    self->nPulseIndex = FIRST_RETURN;
    self->pLasPointFieldsWithExt = NULL;
    self->pExtraPointNativeTypes = NULL;

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
            PyErr_SetString(GETSTATE_FC->error, "BUILD_PULSES must be true or false");    
            return -1;
        }
    }
    
    // only used by las.py but thought it best to store all the options in 
    // this object.
    PyObject *pBinSize = PyDict_GetItemString(pOptionDict, "BIN_SIZE");
    if( pBinSize != NULL )
    {
        PyObject *pBinSizeFloat = PyNumber_Float(pBinSize);
        if( pBinSizeFloat == NULL )
        {
            // exception already set
            return -1;
        }

        self->fBinSize = PyFloat_AsDouble(pBinSizeFloat);
        Py_DECREF(pBinSizeFloat);
    }

    PyObject *pPulseIndex = PyDict_GetItemString(pOptionDict, "PULSE_INDEX");
    if( pPulseIndex != NULL )
    {
        PyObject *pPulseIndexLong = PyNumber_Long(pPulseIndex);
        if( PyLong_Check(pPulseIndexLong) )
        {
            self->nPulseIndex = PyLong_AsLong(pPulseIndexLong);
        }
        else
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "PULSE_INDEX must be an int");    
            return -1;
        }
    }

    LASreadOpener lasreadopener;
    lasreadopener.set_file_name(pszFname);
    self->pReader = lasreadopener.open();

    if( self->pReader == NULL )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Unable to open las file");
        return -1;
    }

    // sets to NULL if no waveforms
    self->pWaveformReader = lasreadopener.open_waveform13(&self->pReader->header);

    return 0;
}

/* calculate the length in case they change in future */
#define GET_LENGTH(x) (sizeof(x) / sizeof(x[0]))

static PyObject *PyLasFileRead_readHeader(PyLasFileRead *self, PyObject *args)
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
    PyDict_SetItemString(pHeaderDict, "FILE_SIGNATURE", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->file_source_ID);
    PyDict_SetItemString(pHeaderDict, "FILE_SOURCE_ID", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->global_encoding);
    PyDict_SetItemString(pHeaderDict, "GLOBAL_ENCODING", pVal);
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

    pVal = PyLong_FromLong(pHeader->version_major);
    PyDict_SetItemString(pHeaderDict, "VERSION_MAJOR", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromLong(pHeader->version_minor);
    PyDict_SetItemString(pHeaderDict, "VERSION_MINOR", pVal);
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

    pVal = PyLong_FromUnsignedLong(pHeader->offset_to_point_data);
    PyDict_SetItemString(pHeaderDict, "OFFSET_TO_POINT_DATA", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->number_of_variable_length_records);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_VARIABLE_LENGTH_RECORDS", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->point_data_format);
    PyDict_SetItemString(pHeaderDict, "POINT_DATA_FORMAT", pVal);
    Py_DECREF(pVal);

    pVal = PyLong_FromUnsignedLong(pHeader->point_data_record_length);
    PyDict_SetItemString(pHeaderDict, "POINT_DATA_RECORD_LENGTH", pVal);
    Py_DECREF(pVal);

    // LAS 1.4
    if( (pHeader->version_major >= 1) && (pHeader->version_minor >= 4 ) )
    {
        pVal = PyLong_FromUnsignedLongLong(pHeader->extended_number_of_point_records);
    }
    else
    {
        pVal = PyLong_FromUnsignedLongLong(pHeader->number_of_point_records);
    }
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_POINT_RECORDS", pVal);
    Py_DECREF(pVal);

    // LAS 1.4
    if( (pHeader->version_major >= 1) && (pHeader->version_minor >= 4 ) )
    {
        pylidar::CVector<U64> number_of_points_by_returnVector(pHeader->extended_number_of_points_by_return,
                            sizeof(pHeader->extended_number_of_points_by_return));
        pVal = (PyObject*)number_of_points_by_returnVector.getNumpyArray(NPY_UINT64);
    }
    else
    {
        pylidar::CVector<U32> number_of_points_by_returnVector(pHeader->number_of_points_by_return, 
                            sizeof(pHeader->number_of_points_by_return));    
        pVal = (PyObject*)number_of_points_by_returnVector.getNumpyArray(NPY_UINT32);
    }
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_POINTS_BY_RETURN", pVal);
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

// helper function
void ConvertCoordsToAngles(double x0, double x1, double y0, double y1, double z0, double z1,
                double *zenith, double *azimuth)
{
    double range = std::sqrt(std::pow(x1 - x0, 2) + 
            std::pow(y1 - y0, 2) + std::pow(z1 - z0, 2));
    *zenith = std::acos((z1 - z0) / range);
    *azimuth = std::atan((x1 - x0) / (y1 - y0));
    if(*azimuth < 0)
    {
       *azimuth = *azimuth + M_PI * 2;
    }                            
}

// read pulses, points, waveforminfo and received for the range.
// it seems only possible to read all these at once with las.
static PyObject *PyLasFileRead_readData(PyLasFileRead *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses = 0;
    if( !PyArg_ParseTuple(args, "|nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    LASpoint *pPoint = &self->pReader->point;

    // start and end pulses optional - only set for non-spatial read
    if( PyTuple_Size(args) == 2 )
    {
        nPulses = nPulseEnd - nPulseStart;
        self->bFinished = false;

        // Can't use self->pReader->seek() to go to an arbitary
        // pulse since this function works on points
        if( nPulseStart < self->nPulsesRead )
        {
            // go back to zero and start again
            self->pReader->seek(0);
            self->nPulsesRead = 0;
            // next if will ignore the needed number of pulses
        }

        if( nPulseStart > self->nPulsesRead )
        {
            // ok now we need to ignore some pulses to get to the right point
            while( self->nPulsesRead < nPulseStart )
            {
                if( !self->pReader->read_point() )
                {
                    // bFinished set below where we can create
                    // empty arrays
                    break;
                }
                // 1-based
                if( !self->bBuildPulses || ( pPoint->return_number == 1 ) )
                    self->nPulsesRead++;
            }
        }
    }
    else if( PyTuple_Size(args) != 0 )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "readData either takes 2 params for non-spatial reads, or 0 params for spatial reads");
        return NULL;

    }

    pylidar::CVector<SLasPulse> pulses(nInitSize, nGrowBy);
    pylidar::CVector<SLasPoint> *pPoints = NULL; // we don't know size - may be extra fields
    pylidar::CVector<SLasWaveformInfo> waveformInfos(nInitSize, nGrowBy);
    pylidar::CVector<U8> received(nInitSize, nGrowBy);
    SLasPulse lasPulse;
    SLasPoint *pLasPoint = NULL; // we don't know size - may be extra fields
    SLasWaveformInfo lasWaveformInfo;
    bool bFinished = false;
    int nReturnNumber = 0; // pPoint->get_return_number can be unreliable so we need another way

    // spatial reads go until there is no more data (see setExtent)
    // non-spatial reads get bFinished updated.
    while( !bFinished )
    {
        if( !self->pReader->read_point() )
        {
            self->bFinished = true;
            // have to do a bit of a hack here since
            // if we are doing spatial reading and building pulses
            // it could be that not all the points were within the 
            // extent so we must check that we haven't got an incomplete
            // set of returns and adjust appropriately.
            if( self->bBuildPulses && (pPoints != NULL))
            {
                SLasPulse *pLastPulse = pulses.getLastElement();
                if( pLastPulse != NULL )
                {
                    pLastPulse->number_of_returns = pPoints->getNumElems() - pLastPulse->pts_start_idx;
                }
            }
            break;
        }

        // check if there are optional extra fields. 
        // it seems best to do this when the reader is all set up.
        // so this should happen once
        if( (pPoint->attributer != NULL) && (self->pLasPointFieldsWithExt == NULL ) )
        {
            // alloc enough mem for standard fields and the extra fields
            int nOldFields = (sizeof(LasPointFields) / sizeof(SpylidarFieldDefn)) - 1; // without sentinel
            int nNewFields = nOldFields + pPoint->attributer->number_attributes + 1; // with sentinel
            self->pLasPointFieldsWithExt = (SpylidarFieldDefn*)malloc(sizeof(SpylidarFieldDefn) * nNewFields);
            // our map of the types
            self->pExtraPointNativeTypes = new std::map<std::string, int>();

            // copy the data over
            memcpy(self->pLasPointFieldsWithExt, LasPointFields, sizeof(SpylidarFieldDefn) * nOldFields);

            // assume extra fields are all floats which seems to make sense as even
            // if they are stored as short etc, there is scaling often applied.
            // TODO: check alignment of location of new fields instead of just
            // using sizeof(SLasPoint) for RISC
            int nNewTotalSize = sizeof(SLasPoint) + 
                    (sizeof(double) * pPoint->attributer->number_attributes);
            // TODO: add any padding required so that the type of the first
            // item in SLasPoint is aligned. This is what C compiler would
            // do (I think)

            // reset the nStructTotalSize field
            for( int i = 0; i < (nNewFields-1); i++ )
            {
                self->pLasPointFieldsWithExt[i].nStructTotalSize = nNewTotalSize;
            }

            // fill in the extra fields
            for( int i = 0; i < pPoint->attributer->number_attributes; i++ )
            {
                SpylidarFieldDefn *pDest = &self->pLasPointFieldsWithExt[nOldFields+i];
                pDest->pszName = pPoint->attributer->attributes[i].name;
                pDest->cKind = 'f';
                pDest->nSize = sizeof(double);
                // TODO: see comment about alignment above for RISC
                pDest->nOffset = sizeof(SLasPoint) + (i * sizeof(double));
                // nStructTotalSize done above

                // now the map of types
                int typenum;
                switch(pPoint->attributer->attributes[i].data_type)
                {
                case LAS_ATTRIBUTE_U8:
                    typenum = NPY_UINT8;
                    break;
                case LAS_ATTRIBUTE_I8:
                    typenum = NPY_INT8;
                    break;
                case LAS_ATTRIBUTE_U16:
                    typenum = NPY_UINT16;
                    break;
                case LAS_ATTRIBUTE_I16:
                    typenum = NPY_INT16;
                    break;
                case LAS_ATTRIBUTE_U32:
                    typenum = NPY_UINT32;
                    break;
                case LAS_ATTRIBUTE_I32:
                    typenum = NPY_INT32;
                    break;
                case LAS_ATTRIBUTE_U64:
                    typenum = NPY_UINT64;
                    break;
                case LAS_ATTRIBUTE_I64:
                    typenum = NPY_INT64;
                    break;
                case LAS_ATTRIBUTE_F32:
                    typenum = NPY_FLOAT32;
                    break;
                case LAS_ATTRIBUTE_F64:
                    typenum = NPY_FLOAT64;
                    break;
                default:
                    fprintf(stderr, "Unkown type for field %s. Assuming f64\n", pPoint->attributer->attributes[i].name);
                    typenum = NPY_FLOAT64;
                }
                self->pExtraPointNativeTypes->insert(std::pair<std::string, int>(pPoint->attributer->attributes[i].name, typenum));
            }

            // ensure sentinel set
            SpylidarFieldDefn *pDest = &self->pLasPointFieldsWithExt[nNewFields-1];
            memset(pDest, 0, sizeof(SpylidarFieldDefn));

            //for( int i = 0; i < nNewFields; i++)
            //{
            //    SpylidarFieldDefn *p = &self->pLasPointFieldsWithExt[i];
            //    fprintf(stderr, "%d %s %c %d %d %d\n", i, p->pszName, p->cKind, p->nSize, p->nOffset, p->nStructTotalSize);
            //}
        }

        // allocate space if needed
        if( pLasPoint == NULL )
        {
            int nSizeStruct = LasPointFields[0].nStructTotalSize;
            if( self->pLasPointFieldsWithExt != NULL )
            {
                // there are extra fields. Use this size instead
                nSizeStruct = self->pLasPointFieldsWithExt[0].nStructTotalSize;
            }
            pLasPoint = (SLasPoint*)malloc(nSizeStruct);

            // now know size of items
            pPoints = new pylidar::CVector<SLasPoint>(nInitSize, nGrowBy, nSizeStruct);
        }

        // always add a new point
        pLasPoint->x = self->pReader->get_x();
        pLasPoint->y = self->pReader->get_y();
        pLasPoint->z = self->pReader->get_z();
        pLasPoint->intensity = pPoint->get_intensity();
        if( pPoint->extended_point_type )
        {
            // use the 'extended' fields since they are bigger
            // I *think* there is no need for the un-extended fields in this case
            pLasPoint->return_number = pPoint->get_extended_return_number() - 1; // 1-based for some reason
            pLasPoint->classification = pPoint->get_extended_classification();
        }
        else
        {
            pLasPoint->return_number = pPoint->get_return_number() - 1; // 1-based for some reason
            pLasPoint->classification = pPoint->get_classification();
        }
        pLasPoint->synthetic_flag = pPoint->get_synthetic_flag();
        pLasPoint->keypoint_flag = pPoint->get_keypoint_flag();
        pLasPoint->withheld_flag = pPoint->get_withheld_flag();
        pLasPoint->user_data = pPoint->get_user_data();
        pLasPoint->point_source_ID = pPoint->get_point_source_ID();
        pLasPoint->deleted_flag = pPoint->get_deleted_flag();
        pLasPoint->extended_point_type = pPoint->extended_point_type; // no function?
        pLasPoint->red = pPoint->rgb[0];        
        pLasPoint->green = pPoint->rgb[1];        
        pLasPoint->blue = pPoint->rgb[2];        
        pLasPoint->nir = pPoint->rgb[3];        

        // now extra fields
        if( self->pLasPointFieldsWithExt != NULL )
        {
            for( I32 i = 0; i < pPoint->attributer->number_attributes; i++ )
            {
                // TODO: _U etc
                double dVal = pPoint->get_attribute_as_float(i);

                // find offset
                SpylidarFieldDefn *pDefn = &self->pLasPointFieldsWithExt[0];
                while( pDefn->pszName != NULL )
                {
                    if( strcmp(pPoint->get_attribute_name(i), pDefn->pszName) == 0 )
                    {
                        // use memcpy so works on SPARC etc without unaligned mem access
                        memcpy((char*)pLasPoint + pDefn->nOffset, &dVal, sizeof(double));
                        break;
                    }
                    pDefn++;
                }
            }
        }

        pPoints->push(pLasPoint);
        //fprintf(stderr, "Pushed new point ret num %d %d\n", pLasPoint->return_number, nReturnNumber);

        // only add a pulse if we are building a pulse per point (self->bBuildPulses == false)
        // or this is the first return of a number of points
        if( !self->bBuildPulses || (nReturnNumber == 0 ) )
        {
            lasPulse.scan_angle_rank = pPoint->get_scan_angle_rank();
            lasPulse.scan_angle = pPoint->get_scan_angle();
            lasPulse.pts_start_idx = pPoints->getNumElems() - 1;
            //fprintf(stderr, "expecting %d points\n", (int)pPoint->get_number_of_returns());
            if( self->bBuildPulses )
                lasPulse.number_of_returns = pPoint->get_number_of_returns();
            else
                lasPulse.number_of_returns = 1;

            // reset - decrement below
            nReturnNumber = lasPulse.number_of_returns;

            lasPulse.orig_number_of_returns = pPoint->get_number_of_returns();
            lasPulse.gps_time = pPoint->get_gps_time();
            lasPulse.scan_direction_flag = pPoint->get_scan_direction_flag();
            lasPulse.edge_of_flight_line = pPoint->get_edge_of_flight_line();
            lasPulse.scanner_channel = pPoint->get_extended_scanner_channel(); // 0 if not 'extended'

            if( self->pWaveformReader != NULL )
            {
                // we have waveforms
                self->pWaveformReader->read_waveform(pPoint);

                U8 lasindex = pPoint->wavepacket.getIndex();
                // if lasindex == 0 then it appears that the wave info insn't available
                // (table 13 in LAS spec)
                if( lasindex > 0 )
                {
                    // fill in the info                
                    lasWaveformInfo.number_of_waveform_received_bins = self->pWaveformReader->nsamples;
                    lasWaveformInfo.received_start_idx = received.getNumElems();
                    lasWaveformInfo.receive_wave_gain = self->pReader->header.vlr_wave_packet_descr[lasindex]->getDigitizerGain();
                    lasWaveformInfo.receive_wave_offset = self->pReader->header.vlr_wave_packet_descr[lasindex]->getDigitizerOffset();

                    /* Get the offset (in ps) from the first digitized value
                    to the location within the waveform packet that the associated 
                    return pulse was detected.*/
                    double location = pPoint->wavepacket.getLocation();

                    // convert to ns
                    lasWaveformInfo.range_to_waveform_start = location*1E3;

                    double pulse_duration = lasWaveformInfo.number_of_waveform_received_bins * 
                        (self->pReader->header.vlr_wave_packet_descr[lasindex]->getTemporalSpacing());

                    lasPulse.number_of_waveform_samples = 1;
                    lasPulse.wfm_start_idx = waveformInfos.getNumElems();

                    waveformInfos.push(&lasWaveformInfo);
                
                    // the actual received data
                    U8 data;
                    for( U32 nCount = 0; nCount < lasWaveformInfo.number_of_waveform_received_bins; nCount++ )
                    {
                        data = self->pWaveformReader->samples[nCount];
                        received.push(&data);
                    }

                    // Set pulse GPS time (ns)
                    lasPulse.gps_time = lasPulse.gps_time - lasWaveformInfo.range_to_waveform_start;

                    // fill in origin, azimuth etc
    
                    /* Set the start location of the return pulse
                       This is calculated as the location of the first return 
                       minus the time offset multiplied by XYZ(t) which is a vector
                       away from the laser origin */
                    lasPulse.x_origin = pLasPoint->x - location * self->pWaveformReader->XYZt[0];
                    lasPulse.y_origin = pLasPoint->y - location * self->pWaveformReader->XYZt[1];
                    lasPulse.z_origin = pLasPoint->z - location * self->pWaveformReader->XYZt[2];

                    /* Get the end location of the return pulse
                      This is calculated as start location of the pulse
                      plus the pulse duration multipled by XYZ(t)
                      It is only used to get the azimuth and zenith angle 
                      of the pulse */
                    double x1 = lasPulse.x_origin + pulse_duration * self->pWaveformReader->XYZt[0];
                    double y1 = lasPulse.y_origin + pulse_duration * self->pWaveformReader->XYZt[1];
                    double z1 = lasPulse.z_origin + pulse_duration * self->pWaveformReader->XYZt[2];
                    ConvertCoordsToAngles(x1, lasPulse.x_origin, y1, lasPulse.y_origin, 
                        z1, lasPulse.z_origin, &lasPulse.zenith, &lasPulse.azimuth);
                }
                else
                {
                    // can't determine origin
                    // might be able to do zenith/azimuth below depending on how many returns
                    lasPulse.x_origin = 0;
                    lasPulse.y_origin = 0;
                    lasPulse.z_origin = 0;
                    lasPulse.zenith = 0;
                    lasPulse.azimuth = 0;
                    lasPulse.number_of_waveform_samples = 0;
                    lasPulse.wfm_start_idx = 0;
                }
            }
            else
            {
                // can't determine origin
                // might be able to do zenith/azimuth below depending on how many returns
                lasPulse.x_origin = 0;
                lasPulse.y_origin = 0;
                lasPulse.z_origin = 0;
                lasPulse.zenith = 0;
                lasPulse.azimuth = 0;
                lasPulse.number_of_waveform_samples = 0;
                lasPulse.wfm_start_idx = 0;
            }

            pulses.push(&lasPulse);
        }

        nReturnNumber--;

        // update loop exit for non-spatial reads
        // spatial reads keep going until all the way through the file
        if( PyTuple_Size(args) == 2 )
        {
            // need to ensure we have read all the points for the last
            // pulse also
            bFinished = (pulses.getNumElems() >= nPulses) && (nReturnNumber == 0);
        }
    }

    self->nPulsesRead += pulses.getNumElems();
    //fprintf(stderr, "pulses %ld points %ld\n", pulses.getNumElems(), pPoints->getNumElems());

    // go through all the pulses and do some tidying up
    for( npy_intp nPulseCount = 0; nPulseCount < pulses.getNumElems(); nPulseCount++)
    {
        SLasPulse *pPulse = pulses.getElem(nPulseCount);
        SLasPoint *p1 = pPoints->getElem(pPulse->pts_start_idx);
        SLasPoint *p2 = NULL;

        // set x_idx and y_idx for the pulses
        if( pPulse->number_of_returns > 0 )
        {
            p2 = pPoints->getElem(pPulse->pts_start_idx + pPulse->number_of_returns - 1);
            if( self->nPulseIndex == FIRST_RETURN )
            {
                pPulse->x_idx = p1->x;
                pPulse->y_idx = p1->y;
            }
            else
            {
                pPulse->x_idx = p2->x;
                pPulse->y_idx = p2->y;
            }
        }

        // find those with 
        // number_of_returns > 1 and use point locations to fill in 
        // zenith, azimuth etc if self->bBuildPulses
        if( self->bBuildPulses && ( pPulse->number_of_returns > 1 ) && ( pPulse->zenith == 0 ) && ( pPulse->azimuth == 0) )
        {
            ConvertCoordsToAngles(p2->x, p1->x, p2->y, p1->y, p2->z, p1->z,
                        &pPulse->zenith, &pPulse->azimuth);
        }
    }

    // as we only allocated when we knew the size with extra fields
    free(pLasPoint);

    PyArrayObject *pNumpyPulses = pulses.getNumpyArray(LasPulseFields);
    SpylidarFieldDefn *pPointDefn = LasPointFields;
    if( self->pLasPointFieldsWithExt != NULL )
    {
        // extra fields - use other definition instead
        pPointDefn = self->pLasPointFieldsWithExt;
    }

    if( pPoints == NULL )
    {
        // There were no points loaded in this call. 
        // We still need to create an empty array so go through the process
        // so the fields match any arrays we have already returned.
        int nSizeStruct = LasPointFields[0].nStructTotalSize;
        if( self->pLasPointFieldsWithExt != NULL )
        {
            // there are extra fields. Use this size instead
            nSizeStruct = self->pLasPointFieldsWithExt[0].nStructTotalSize;
        }
        // now know size of items
        pPoints = new pylidar::CVector<SLasPoint>(nInitSize, nGrowBy, nSizeStruct);
    }

    PyArrayObject *pNumpyPoints = pPoints->getNumpyArray(pPointDefn);
    delete pPoints;
    PyArrayObject *pNumpyInfos = waveformInfos.getNumpyArray(LasWaveformInfoFields);
    PyArrayObject *pNumpyReceived = received.getNumpyArray(NPY_UINT8);

    // build tuple
    PyObject *pTuple = PyTuple_Pack(4, pNumpyPulses, pNumpyPoints, pNumpyInfos, pNumpyReceived);

    // decref the objects since we have finished with them (PyTuple_Pack increfs)
    Py_DECREF(pNumpyPulses);
    Py_DECREF(pNumpyPoints);
    Py_DECREF(pNumpyInfos);
    Py_DECREF(pNumpyReceived);

    return pTuple;
}

static PyObject *PyLasFileRead_getEPSG(PyLasFileRead *self, PyObject *args)
{
    /** Taken from SPDlib
    Get EPSG projection code from LAS file header
         
    TODO: Needs testing with a range of coordinate systems. Within lasinfo a number of 
    checks for differnent keys are used. Need to confirm only checking for key id 3072 is
    sufficient.
         
    */
    bool foundProjection = false;
    long nEPSG = 0;

    LASheader *pHeader = &self->pReader->header;
    if( pHeader->vlr_geo_keys != NULL )
    {
        for (int j = 0; j < pHeader->vlr_geo_keys->number_of_keys; j++)
        {
            if(pHeader->vlr_geo_key_entries[j].key_id == 3072)
            {
                nEPSG = pHeader->vlr_geo_key_entries[j].value_offset;
                foundProjection = true;
            }
        }
    }

    if( !foundProjection )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Cannot find EPSG code for Coordinate System");
        return NULL;
    }
    
    return PyLong_FromLong(nEPSG);
}

static PyObject *PyLasFileRead_setExtent(PyLasFileRead *self, PyObject *args)
{
    double xMin, xMax, yMin, yMax;
    if( !PyArg_ParseTuple(args, "dddd:setExtent", &xMin, &xMax, &yMin, &yMax ) )
        return NULL;

    // set the new extent - point should now only be within these coords
    self->pReader->inside_rectangle(xMin, yMin, xMax, yMax);
    // seek back to the start - laslib doesn't seem to do this
    // we want to read all points in the file within these coords
    self->pReader->seek(0);

    Py_RETURN_NONE;
}

static PyObject *PyLasFileRead_getScaling(PyLasFileRead *self, PyObject *args)
{
    const char *pszField;
    if( !PyArg_ParseTuple(args, "s:getScaling", &pszField ) )
        return NULL;

    double dGain = 1.0, dOffset = 0;
    LASheader *pHeader = &self->pReader->header;
    if( strcmp(pszField, "X") == 0 )
    {
        dGain = 1.0 / pHeader->x_scale_factor;
        dOffset = pHeader->x_offset;
    }
    else if( strcmp(pszField, "Y") == 0 )
    {
        dGain = 1.0 / pHeader->y_scale_factor;
        dOffset = pHeader->y_offset;
    }
    else if( strcmp(pszField, "Z") == 0 )
    {
        dGain = 1.0 / pHeader->z_scale_factor;
        dOffset = pHeader->z_offset;
    }
    else
    {
        // no luck, try the attributes
        bool bFound = false;
        for( I32 index = 0; index < pHeader->number_attributes; index++ )
        {
            LASattribute *pAttr = &pHeader->attributes[index];
            if( strcmp(pszField, pAttr->name) == 0)
            {
                dGain = pAttr->scale[0];
                dOffset = pAttr->offset[0];
                bFound = true;
                break;
            }
        }

        if( !bFound )
        {
            // raise Python exception
            PyErr_Format(GETSTATE_FC->error, "Unable to get scaling for field %s", pszField);
            return NULL;
        }
    }

    return Py_BuildValue("dd", dGain, dOffset);
}

// i've implemented this for read and write - not sure how much use it is
static PyObject *PyLasFileRead_getNativeDataType(PyLasFileRead *self, PyObject *args)
{
    const char *pszField;
    if( !PyArg_ParseTuple(args, "s:getNativeDataType", &pszField) )
        return NULL;

    PyArray_Descr *pDescr = NULL;
    // X, Y and Z are always int32 - we just present them as double
    // the other fields mirror the underlying type
    if( ( strcmp(pszField, "X") == 0) || ( strcmp(pszField, "Y") == 0) ||
        ( strcmp(pszField, "Z") == 0) )
    {
        pDescr = PyArray_DescrFromType(NPY_INT32);
    }
    else
    {
        // see if it is one of the extra fields
        if( self->pExtraPointNativeTypes != NULL )
        {
            std::map<std::string, int>::iterator itr;
            itr = self->pExtraPointNativeTypes->find(pszField);
            if( itr != self->pExtraPointNativeTypes->end() )
            {
                pDescr = PyArray_DescrFromType(itr->second);
            }
        }

        // if not, it should be one of the standard fields we create
        if( pDescr == NULL )
        {
            pDescr = pylidar_getDtypeForField(LasPointFields, pszField);
        }

        if( pDescr == NULL )
        {
            // raise Python exception
            PyErr_Format(GETSTATE_FC->error, "Unable to find data type for %s", pszField);
            return NULL;
        }
    }    

    // ok turns out that a 'Descr' is different to the Python 'type' object
    // however it does have a field that points to a type object
    PyTypeObject *pTypeObj = pDescr->typeobj;

    // increment this refcount, but don't need the whole structure now
    // so decref
    Py_INCREF(pTypeObj);
    Py_DECREF(pDescr);

    return (PyObject*)pTypeObj;
}

/* Table of methods */
static PyMethodDef PyLasFileRead_methods[] = {
    {"readHeader", (PyCFunction)PyLasFileRead_readHeader, METH_NOARGS, NULL},
    {"readData", (PyCFunction)PyLasFileRead_readData, METH_VARARGS, NULL}, 
    {"getEPSG", (PyCFunction)PyLasFileRead_getEPSG, METH_NOARGS, NULL},
    {"setExtent", (PyCFunction)PyLasFileRead_setExtent, METH_VARARGS, NULL},
    {"getScaling", (PyCFunction)PyLasFileRead_getScaling, METH_VARARGS, NULL},
    {"getNativeDataType", (PyCFunction)PyLasFileRead_getNativeDataType, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyObject *PyLasFileRead_getBuildPulses(PyLasFileRead *self, void *closure)
{
    if( self->bBuildPulses )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyLasFileRead_getHasSpatialIndex(PyLasFileRead *self, void *closure)
{
    if( self->pReader->get_index() != NULL )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyLasFileRead_getFinished(PyLasFileRead *self, void *closure)
{
    if( self->bFinished )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyLasFileRead_getPulsesRead(PyLasFileRead *self, void *closure)
{
    return PyLong_FromSsize_t(self->nPulsesRead);
}

static PyObject *PyLasFileRead_getBinSize(PyLasFileRead *self, void *closure)
{
    return PyFloat_FromDouble(self->fBinSize);
}

static int PyLasFileRead_setBinSize(PyLasFileRead *self, PyObject *value, void *closure)
{
    PyObject *pBinSizeFloat = PyNumber_Float(value);
    if( pBinSizeFloat == NULL )
    {
        // exception already set
        return -1;
    }

    self->fBinSize = PyFloat_AsDouble(pBinSizeFloat);
    Py_DECREF(pBinSizeFloat);
    return 0;
}


/* get/set */
static PyGetSetDef PyLasFileRead_getseters[] = {
    {(char*)"build_pulses", (getter)PyLasFileRead_getBuildPulses, NULL, 
        (char*)"Whether we are building pulses of multiple points when reading", NULL},
    {(char*)"hasSpatialIndex", (getter)PyLasFileRead_getHasSpatialIndex, NULL,
        (char*)"Whether a spatial index exists for this file", NULL},
    {(char*)"finished", (getter)PyLasFileRead_getFinished, NULL, 
        (char*)"Whether we have finished reading the file or not", NULL},
    {(char*)"pulsesRead", (getter)PyLasFileRead_getPulsesRead, NULL,
        (char*)"Number of pulses read", NULL},
    {(char*)"binSize", (getter)PyLasFileRead_getBinSize, (setter)PyLasFileRead_setBinSize,
        (char*)"Bin size to use for spatial data", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLasFileReadType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_las.FileRead",         /*tp_name*/
    sizeof(PyLasFileRead),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyLasFileRead_dealloc, /*tp_dealloc*/
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
    "Las File Reader object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyLasFileRead_methods,             /* tp_methods */
    0,             /* tp_members */
    PyLasFileRead_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyLasFileRead_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

/* Python object wrapping a LASwriter */
typedef struct {
    PyObject_HEAD
    // set by _init so we can create LASwriteOpener
    // when writing the first block
    char *pszFilename;
    // created when writing first block
    LASwriter *pWriter;
    LASheader *pHeader;
    LASpoint *pPoint;
    LASwaveform13writer *pWaveformWriter;
    // set by setEPSG
    int nEPSG;
    // X, Y and Z scaling
    // set by setScaling
    bool bXScalingSet;
    bool bYScalingSet;
    bool bZScalingSet;
    F64 dXGain;
    F64 dXOffset;
    F64 dYGain;
    F64 dYOffset;
    F64 dZGain;
    F64 dZOffset;
    // driver options
    U8 point_data_format;
    U16 point_data_record_length;
    // following need to be set up before first block written
    // or they will be ignored.
    // set by setScaling for 'attribute' fields
    std::map<std::string, std::pair<double, double> > *pScalingMap;
    // set by setNativeDataType
    std::map<std::string, pylidar::SFieldInfo> *pAttributeTypeMap;
    // set by WAVEFORM_DESCR driver option
    PyArrayObject *pWaveformDescr;
} PyLasFileWrite;


/* destructor - close and delete */
static void 
PyLasFileWrite_dealloc(PyLasFileWrite *self)
{
    if(self->pszFilename != NULL)
    {
        free(self->pszFilename);
    }
    if(self->pWaveformWriter != NULL)
    {
        self->pWaveformWriter->close();
        delete self->pWaveformWriter;
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
    if(self->pPoint != NULL)
    {
        delete self->pPoint;
    }
    if(self->pScalingMap != NULL)
    {
        delete self->pScalingMap;
    }
    if(self->pAttributeTypeMap != NULL)
    {
        delete self->pAttributeTypeMap;
    }
    Py_XDECREF(self->pWaveformDescr);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* init method - open file */
static int 
PyLasFileWrite_init(PyLasFileWrite *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;
PyObject *pOptionDict;

    if( !PyArg_ParseTuple(args, "sO", &pszFname, &pOptionDict ) )
    {
        return -1;
    }

    // defaults from lasexample_write_only.cpp
    // I can't find any documentation so I assume these
    // are sensible.
    self->point_data_format = 1;
    self->point_data_record_length = 28;
    self->pWaveformDescr = NULL;
    if( pOptionDict != Py_None )
    {
        if( !PyDict_Check(pOptionDict) )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "Last parameter to init function must be a dictionary");
            return -1;
        }

        PyObject *pVal = PyDict_GetItemString(pOptionDict, "FORMAT_VERSION");
        if( pVal != NULL )
        {
            if( !PyLong_Check(pVal) )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "FORMAT parameter must be integer");
                return -1;
            }
            self->point_data_format = PyLong_AsLong(pVal);
        }

        pVal = PyDict_GetItemString(pOptionDict, "RECORD_LENGTH");
        if( pVal != NULL )
        {
            if( !PyLong_Check(pVal) )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "RECORD_LENGTH parameter must be integer");
                return -1;
            }
            self->point_data_record_length = PyLong_AsLong(pVal);
        }

        pVal = PyDict_GetItemString(pOptionDict, "WAVEFORM_DESCR");
        if( pVal != NULL )
        {
            if( !PyArray_Check(pVal) )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "WAVEFORM_DESCR parameter must be an array");
                return -1;
            }
            if( PyArray_SIZE((PyArrayObject*)pVal) > 256 )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "WAVEFORM_DESCR parameter must be shorter than 256 - LAS restriction");
                return -1;
            }
            Py_INCREF(pVal);
            self->pWaveformDescr = (PyArrayObject*)pVal;
        }
    }

    // set up when first block written
    // so user can update the header
    self->pWriter = NULL;
    self->pHeader = NULL;
    self->pPoint = NULL;
    self->pWaveformWriter = NULL;
    self->nEPSG = 0;
    self->dXGain = 0;
    self->dXOffset = 0;
    self->dYGain = 0;
    self->dYOffset = 0;
    self->dZGain = 0;
    self->dZOffset = 0;
    self->bXScalingSet = false;
    self->bYScalingSet = false;
    self->bZScalingSet = false;
    self->pScalingMap = new std::map<std::string, std::pair<double, double> >;
    self->pAttributeTypeMap = new std::map<std::string, pylidar::SFieldInfo>;

    // copy filename so we can open later
    self->pszFilename = strdup(pszFname);

    return 0;
}

// copies recognised fields from pHeaderDict into pHeader
void setHeaderFromDictionary(PyObject *pHeaderDict, LASheader *pHeader)
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

    pVal = PyDict_GetItemString(pHeaderDict, "FILE_SOURCE_ID");
    if( pVal != NULL )
        pHeader->file_source_ID = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "GLOBAL_ENCODING");
    if( pVal != NULL )
        pHeader->global_encoding = PyLong_AsLong(pVal);

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

    pVal = PyDict_GetItemString(pHeaderDict, "VERSION_MAJOR");
    if( pVal != NULL )
        pHeader->version_major = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "VERSION_MINOR");
    if( pVal != NULL )
        pHeader->version_minor = PyLong_AsLong(pVal);

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

    pVal = PyDict_GetItemString(pHeaderDict, "OFFSET_TO_POINT_DATA");
    if( pVal != NULL )
        pHeader->offset_to_point_data = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "NUMBER_OF_VARIABLE_LENGTH_RECORDS");
    if( pVal != NULL )
        pHeader->number_of_variable_length_records = PyLong_AsLong(pVal);

    if( pVal != NULL )
        pHeader->point_data_record_length = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "NUMBER_OF_POINT_RECORDS");
    if( pVal != NULL )
        pHeader->number_of_point_records = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "NUMBER_OF_POINTS_BY_RETURN");
    if( pVal != NULL )
    {    
        PyObject *pArray = PyArray_FROM_OT(pVal, NPY_UINT8);
        // TODO: check 1d?
        for( npy_intp i = 0; (i < PyArray_DIM((PyArrayObject*)pArray, 0)) && 
                            (i < (npy_intp)GET_LENGTH(pHeader->number_of_points_by_return)); i++ )
        {
            pHeader->number_of_points_by_return[i] = *((U8*)PyArray_GETPTR1((PyArrayObject*)pArray, i));
        }
        Py_DECREF(pArray);
    }

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

// sets the vlr_wave_packet_descr field on the header from the array
// returned by las.getWavePacketDescriptions()
void setWavePacketDescr(PyArrayObject *pArray, LASheader *pHeader, U8 nBitsPerSample)
{
    npy_intp nSize = PyArray_SIZE(pArray);
    if( nSize > 0 )
    {
        pylidar::CFieldInfoMap infoMap(pArray);
        // ok apparently there are always 256. _init raises an error if more.
        pHeader->vlr_wave_packet_descr = new LASvlr_wave_packet_descr*[256];
        memset(pHeader->vlr_wave_packet_descr, 0, sizeof(LASvlr_wave_packet_descr*) * 256);
        for( npy_intp n = 0; n < nSize; n++ )
        {
            void *pRow = PyArray_GETPTR1(pArray, n);
            LASvlr_wave_packet_descr *pDescr = new LASvlr_wave_packet_descr;
            pDescr->setNumberOfSamples(infoMap.getIntValue(N_WAVEFORM_BINS, pRow));
            pDescr->setDigitizerGain(infoMap.getDoubleValue(RECEIVE_WAVE_GAIN, pRow));
            pDescr->setDigitizerOffset(infoMap.getDoubleValue(RECEIVE_WAVE_OFFSET, pRow));
            pDescr->setBitsPerSample(nBitsPerSample);
            pHeader->vlr_wave_packet_descr[n] = pDescr;
        }
    }
}

// gets the index info pHeader->vlr_wave_packet_descr
// that is needed by LASwavepacket.setIndex
U8 getWavepacketIndex(LASheader *pHeader, U32 nSamples, F64 fGain, F64 fOffset)
{
    U8 index = 0; // not sure what to do on error
    for( int n = 0; n < 256; n++ )
    {
        LASvlr_wave_packet_descr *pDescr = pHeader->vlr_wave_packet_descr[n];
        if( pDescr != NULL )
        {
            if( (pDescr->getNumberOfSamples() == nSamples) && (pDescr->getDigitizerGain() == fGain)
                && (pDescr->getDigitizerOffset() == fOffset) )
            {
                index = n;
                break;
            }
        }
    }
    return index;
}

// returns a set of names we recognise and match to 
// 'essential' fields in the LASpoint structure
std::set<std::string> getEssentialPointFieldNames()
{
    std::set<std::string> nonAttrPointSet;
    SpylidarFieldDefn *pDefn = LasPointFields;
    while(pDefn->pszName != NULL )
    {
        /* Convert to upper case */
        char *pszName = strdup(pDefn->pszName);
        for( int i = 0; pszName[i] != '\0'; i++ )
        {
            pszName[i] = toupper(pszName[i]);
        }

        nonAttrPointSet.insert(pszName);

        free(pszName);
        pDefn++;
    }

    return nonAttrPointSet;
}

static PyObject *PyLasFileWrite_writeData(PyLasFileWrite *self, PyObject *args)
{
    PyObject *pHeader, *pPulses, *pPoints, *pWaveformInfos, *pReceived;
    if( !PyArg_ParseTuple(args, "OOOOO:writeData", &pHeader, &pPulses, &pPoints, &pWaveformInfos, &pReceived ) )
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
    if( bArraysOk && (bHaveWaveformInfos != bHaveReceived) )
    {
        bArraysOk = false;
        pszMessage = "Either both waveform info and reveived must be set, or neither";
    }
    if( bArraysOk && bHaveWaveformInfos && !PyArray_Check(pWaveformInfos) )
    {
        bArraysOk = false;
        pszMessage = "Waveform info must be a numpy array";
    }
    if( bArraysOk && bHaveReceived && !PyArray_Check(pReceived) )
    {
        bArraysOk = false;
        pszMessage = "transmitted must be a numpy array";
    }
    if( bArraysOk && ((PyArray_NDIM((PyArrayObject*)pPulses) != 1) || (PyArray_NDIM((PyArrayObject*)pPoints) != 2) || 
            (bHaveWaveformInfos && (PyArray_NDIM((PyArrayObject*)pWaveformInfos) != 2)) || 
            (bHaveReceived && (PyArray_NDIM((PyArrayObject*)pReceived) != 3)) ) )
    {
        bArraysOk = false;
        pszMessage = "pulses must be 1d, points and waveforminfo 2d, received 3d";
    }
    if( bArraysOk && bHaveReceived && (PyArray_TYPE((PyArrayObject*)pReceived) != NPY_UINT16))
    {
        // uint16 set by las.py
        bArraysOk = false;
        pszMessage = "received must be 16bit";
    }

    if( !bArraysOk )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, pszMessage);
        return NULL;
    }

    // create a mapping between names of fields and their 'info' which has
    // the type, offset etc
    // do it up here in case we need info for the attributes when setting
    // up the header and writer
    pylidar::CFieldInfoMap pulseMap((PyArrayObject*)pPulses);
    pylidar::CFieldInfoMap pointMap((PyArrayObject*)pPoints);

    if( self->pWriter == NULL )
    {
        // check that all the scaling has been set
        if( !self->bXScalingSet || !self->bYScalingSet || !self->bZScalingSet )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "Must set scaling for X, Y and Z columns before writing data");
            return NULL;
        }

        // create writer
        self->pHeader = new LASheader;
        self->pPoint = new LASpoint;

        // set scaling
        self->pHeader->x_scale_factor = 1.0 / self->dXGain;
        self->pHeader->y_scale_factor = 1.0 / self->dYGain;
        self->pHeader->z_scale_factor = 1.0 / self->dZGain;
        self->pHeader->x_offset = self->dXOffset;
        self->pHeader->y_offset = self->dYOffset;
        self->pHeader->z_offset = self->dZOffset;
        
        // populate header from pHeader dictionary
        if( pHeader != Py_None )
        {
            setHeaderFromDictionary(pHeader, self->pHeader);
        }

        // set epsg
        if( self->nEPSG != 0 )
        {
            LASvlr_key_entry ent;
            ent.key_id = 3072;
            ent.value_offset = self->nEPSG;
            ent.tiff_tag_location = 0;
            ent.count = 1;
            self->pHeader->set_geo_keys(1, &ent);
        }

        // work out what are the additional (point) fields that aren't standard LAS
        // ones. 'attributes' in laslib terms.
        // temp set of the fields we use to populate LASpoint fields
        // others must be attributes
        std::set<std::string> nonAttrPointSet = getEssentialPointFieldNames();

        // iterate over our field map and see what isn't in nonAttrPointSet
        for( std::map<std::string, pylidar::SFieldInfo>::iterator itr = pointMap.begin(); itr != pointMap.end(); itr++ )
        {
            if( nonAttrPointSet.find(itr->first) == nonAttrPointSet.end() )
            {
                // not in our 'compulsory' fields - must be an attribute
                char cKind = itr->second.cKind;
                int nSize = itr->second.nSize;

                // have they requested a different type?
                std::map<std::string, pylidar::SFieldInfo>::iterator attrTypeItr = self->pAttributeTypeMap->find(itr->first);
                if( attrTypeItr != self->pAttributeTypeMap->end() )
                {
                    cKind = attrTypeItr->second.cKind;
                    nSize = attrTypeItr->second.nSize;
                }

                // tell the header
                U32 type = 0;
                if( (cKind == 'u') && (nSize == 1) )
                    type = LAS_ATTRIBUTE_U8;
                else if( (cKind == 'i') && (nSize == 1) )
                    type = LAS_ATTRIBUTE_I8;
                else if( (cKind == 'u') && (nSize == 2) )
                    type = LAS_ATTRIBUTE_U16;
                else if( (cKind == 'i') && (nSize == 2) )
                    type = LAS_ATTRIBUTE_I16;
                else if( (cKind == 'u') && (nSize == 4) )
                    type = LAS_ATTRIBUTE_U32;
                else if( (cKind == 'i') && (nSize == 4) )
                    type = LAS_ATTRIBUTE_I32;
                else if( (cKind == 'u') && (nSize == 8) )
                    type = LAS_ATTRIBUTE_U64;
                else if( (cKind == 'i') && (nSize == 8) )
                    type = LAS_ATTRIBUTE_I64;
                else if( (cKind == 'f') && (nSize == 4) )
                    type = LAS_ATTRIBUTE_F32;
                else if( (cKind == 'f') && (nSize == 8) )
                    type = LAS_ATTRIBUTE_F64;
                // else error?

                LASattribute attr(type, itr->first.c_str());
                // have they set the scaling for this field?
                std::map<std::string, std::pair<double, double> >::iterator scalingItr = self->pScalingMap->find(itr->first);
                if( scalingItr != self->pScalingMap->end() )
                {
                    std::pair<double, double> vals = scalingItr->second;
                    attr.set_scale(vals.first);
                    attr.set_offset(vals.second);
                }
                // otherwise laslib sets 1.0, 0 etc

                self->pHeader->add_attribute(attr);
            }
        }

        // add a vlr if we actually have attributes
        // note - I don't completely understand how this all works.
        if( self->pHeader->number_attributes > 0 )
        {
            self->pHeader->add_vlr("LASF_Spec\0\0\0\0\0\0", 4, self->pHeader->number_attributes*sizeof(LASattribute),
                    (U8*)self->pHeader->attributes, FALSE, "by Pylidar");
        }

        // do we have waveform info?
        if( ( self->pWaveformDescr != NULL ) && bHaveWaveformInfos && bHaveReceived)
        {
            U8 nBitsPerSample = PyArray_ITEMSIZE((PyArrayObject*)pReceived) * 8;
            setWavePacketDescr(self->pWaveformDescr, self->pHeader, nBitsPerSample);
        }

        // point_data_format and point_data_record_length set in init
        // from option dict if available
        self->pHeader->point_data_format = self->point_data_format;
        self->pHeader->point_data_record_length = self->point_data_record_length;
        self->pPoint->init(self->pHeader, self->point_data_format, 
                self->point_data_record_length, self->pHeader);

        // need to do this after pPoint->init since that deletes extra_bytes
        // not sure why we have to do this at all, but there you go.
        self->pPoint->extra_bytes = new U8[self->pHeader->get_attributes_size()];

        LASwriteOpener laswriteopener;
        laswriteopener.set_file_name(self->pszFilename);
        
        self->pWriter = laswriteopener.open(self->pHeader);
        if( self->pWriter == NULL )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "Unable to open las file");
            return NULL;
        }

        // waveform writer?
        // set by setWavePacketDescr
        if( self->pHeader->vlr_wave_packet_descr != NULL )
        {
            self->pWaveformWriter = new LASwaveform13writer;
            if( !self->pWaveformWriter->open(self->pszFilename, self->pHeader->vlr_wave_packet_descr) )
            {
                // raise Python exception
                PyErr_SetString(GETSTATE_FC->error, "Unable to open las waveform file");
                return NULL;
            }
        }
    }

    // now write all the pulses
    for( npy_intp nPulseIdx = 0; nPulseIdx < PyArray_DIM((PyArrayObject*)pPulses, 0); nPulseIdx++)
    {
        void *pPulseRow = PyArray_GETPTR1((PyArrayObject*)pPulses, nPulseIdx);
        // fill in the info from the pulses
        npy_int64 nPoints = pulseMap.getIntValue("NUMBER_OF_RETURNS", pPulseRow);

        self->pPoint->set_scan_angle_rank(pulseMap.getDoubleValue("SCAN_ANGLE_RANK", pPulseRow));
        // TODO: check if extended?
        self->pPoint->set_extended_scan_angle(pulseMap.getDoubleValue("SCAN_ANGLE", pPulseRow));
        self->pPoint->set_number_of_returns(nPoints);
        // TODO: GPS_TIME or TIMESTAMP?
        self->pPoint->set_gps_time(pulseMap.getDoubleValue("GPS_TIME", pPulseRow));
        self->pPoint->set_scan_direction_flag(pulseMap.getIntValue("SCAN_DIRECTION_FLAG", pPulseRow));
        self->pPoint->set_edge_of_flight_line(pulseMap.getIntValue("EDGE_OF_FLIGHT_LINE", pPulseRow));
        self->pPoint->set_extended_scanner_channel(pulseMap.getIntValue("SCANNER_CHANNEL", pPulseRow));

        // now the point
        for( npy_intp nPointCount = 0; nPointCount < nPoints; nPointCount++ )
        {
            void *pPointRow = PyArray_GETPTR2((PyArrayObject*)pPoints, nPointCount, nPulseIdx);
            // TODO: extended creation option?
            npy_int64 nExtended = pointMap.getIntValue("EXTENDED_POINT_TYPE", pPointRow);
            self->pPoint->extended_point_type = nExtended;
            self->pPoint->set_x(pointMap.getDoubleValue("X", pPointRow));
            self->pPoint->set_y(pointMap.getDoubleValue("Y", pPointRow));
            self->pPoint->set_z(pointMap.getDoubleValue("Z", pPointRow));
            self->pPoint->set_intensity(pointMap.getIntValue("INTENSITY", pPointRow));
            
            npy_int64 nClassification = pointMap.getIntValue("CLASSIFICATION", pPointRow);
            if(nExtended)
            {
                self->pPoint->extended_return_number = nPointCount + 1;
                self->pPoint->set_extended_classification(nClassification);
            }
            else
            {
                self->pPoint->set_return_number(nPointCount + 1);
                self->pPoint->set_classification(nClassification);
            }
            
            self->pPoint->set_synthetic_flag(pointMap.getIntValue("SYNTHETIC_FLAG", pPointRow));
            self->pPoint->set_keypoint_flag(pointMap.getIntValue("KEYPOINT_FLAG", pPointRow));
            self->pPoint->set_withheld_flag(pointMap.getIntValue("WITHHELD_FLAG", pPointRow));
            self->pPoint->set_user_data(pointMap.getIntValue("USER_DATA", pPointRow));
            self->pPoint->set_point_source_ID(pointMap.getIntValue("POINT_SOURCE_ID", pPointRow));
            self->pPoint->set_deleted_flag(pointMap.getIntValue("DELETED_FLAG", pPointRow));
            self->pPoint->rgb[0] = pointMap.getIntValue("RED", pPointRow);
            self->pPoint->rgb[1] = pointMap.getIntValue("GREEN", pPointRow);
            self->pPoint->rgb[2] = pointMap.getIntValue("BLUE", pPointRow);
            self->pPoint->rgb[3] = pointMap.getIntValue("NIR", pPointRow);

            // now loop through any attributes and set them
            for( I32 index = 0; index < self->pHeader->number_attributes; index++ )
            {
                LASattribute *pAttr = &self->pPoint->attributer->attributes[index];
                double dVal = pointMap.getDoubleValue(pAttr->name, pPointRow);
                // apply scaling - is a no-op if not set since scale, offset is 1, 0
                dVal = (dVal / pAttr->scale[0]) - pAttr->offset[0];
                // convert dVal back to the type that we write it as
                // there is a method to do this, but bizarrely made private
                I32 type = ((I32)pAttr->data_type - 1)%10;
                switch(type)
                {
                    case LAS_ATTRIBUTE_U8:
                    {
                        U8 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_I8:
                    {
                        I8 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_U16:
                    {
                        U16 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_I16:
                    {
                        I16 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_U32:
                    {
                        U32 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_I32:
                    {
                        I32 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_U64:
                    {
                        U64 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_I64:
                    {
                        I64 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_F32:
                    {
                        F32 val = dVal;
                        self->pPoint->set_attribute(index, (U8*)&val);
                    }
                    break;
                    case LAS_ATTRIBUTE_F64:
                    self->pPoint->set_attribute(index, (U8*)&dVal);
                    break;
                }
                index++;
            }

            // now waveforms
            self->pPoint->have_wavepacket = FALSE;
            if( bHaveWaveformInfos && bHaveReceived && (self->pWaveformWriter != NULL) )
            {
                // set point data
                // note that this will be repeated for each point that is part
                // of the pulse
                npy_int64 nInfos = pulseMap.getIntValue("NUMBER_OF_WAVEFORM_SAMPLES", pPulseRow);
                if( nInfos > 0 ) // print error if more than 1? LAS can only handle 1
                {
                    pylidar::CFieldInfoMap waveMap((PyArrayObject*)pWaveformInfos); // create once?
                    void *pInfoRow = PyArray_GETPTR2((PyArrayObject*)pWaveformInfos, 0, nPulseIdx);
                    U32 nSamples = waveMap.getIntValue(N_WAVEFORM_BINS, pInfoRow);
                    F64 fGain = waveMap.getDoubleValue(RECEIVE_WAVE_GAIN, pInfoRow);
                    F64 fOffset = waveMap.getDoubleValue(RECEIVE_WAVE_OFFSET, pInfoRow);
                    U8 index = getWavepacketIndex(self->pHeader, nSamples, fGain, fOffset);

                    self->pPoint->wavepacket.setIndex(index);
                    self->pPoint->have_wavepacket = TRUE;

                    U16 *pBuffer = (U16*)malloc(sizeof(U16) * nSamples);
                    for( U32 n = 0; n < nSamples; n++ )
                    {
                        void *pSample = PyArray_GETPTR3((PyArrayObject*)pReceived, n, 0, nPulseIdx);
                        memcpy(&pBuffer[n], pSample, sizeof(U16));
                    }
                    self->pWaveformWriter->write_waveform(self->pPoint, (U8*)pBuffer);
                    free(pBuffer);
                    // TODO: zentih, azimuth -> vector etc
                }

            }

            self->pWriter->write_point(self->pPoint);
            self->pWriter->update_inventory(self->pPoint);
        }

        
    }

    Py_RETURN_NONE;
}

static PyObject *PyLasFileWrite_setEPSG(PyLasFileWrite *self, PyObject *args)
{
    // just grab the epsg for now. Written when we write the header.
    if( !PyArg_ParseTuple(args, "i:setEPSG", &self->nEPSG ) )
        return NULL;
    
    Py_RETURN_NONE;
}

static PyObject *PyLasFileWrite_setScaling(PyLasFileWrite *self, PyObject *args)
{
    double dGain, dOffset;
    const char *pszField;
    if( !PyArg_ParseTuple(args, "sdd:setScaling", &pszField, &dGain, &dOffset) )
        return NULL;

    // note: only handle X, Y and Z on the points at the moment
    // potentially this scheme could be extended to 'additional' fields
    // ie las attributes. 
    if( strcmp(pszField, "X") == 0)
    {
        self->dXGain = dGain;
        self->dXOffset = dOffset;
        self->bXScalingSet = true;
    }
    else if( strcmp(pszField, "Y") == 0)
    {
        self->dYGain = dGain;
        self->dYOffset = dOffset;
        self->bYScalingSet = true;
    }
    else if( strcmp(pszField, "Z") == 0)
    {
        self->dZGain = dGain;
        self->dZOffset = dOffset;
        self->bZScalingSet = true;
    }
    else
    {
        std::set<std::string> essentialFields = getEssentialPointFieldNames();
        if( essentialFields.find(pszField) != essentialFields.end() )
        {
            // is an essential field that we can't store scaling for.
            // raise Python exception
            PyErr_Format(GETSTATE_FC->error, "Unable to set scaling for field %s", pszField);
            return NULL;
        }
        else
        {
            // store for creation
            self->pScalingMap->insert(std::pair<std::string, std::pair<double, double> >(pszField, std::pair<double, double>(dGain, dOffset)));
        }
    }

    Py_RETURN_NONE;
}

static PyObject *PyLasFileWrite_getNativeDataType(PyLasFileWrite *self, PyObject *args)
{
    const char *pszField;
    if( !PyArg_ParseTuple(args, "s:getNativeDataType", &pszField) )
        return NULL;

    PyArray_Descr *pDescr = NULL;
    // X, Y and Z are always int32 - we just present them as double
    // the other fields mirror the underlying type
    if( ( strcmp(pszField, "X") == 0) || ( strcmp(pszField, "Y") == 0) ||
        ( strcmp(pszField, "Z") == 0) )
    {
        pDescr = PyArray_DescrFromType(NPY_INT32);
    }
    else
    {
        // first check the essential fields
        pDescr = pylidar_getDtypeForField(LasPointFields, pszField);
        if( ( pDescr == NULL ) && (self->pHeader == NULL) )
        {
            // no luck, try the attributes that have been set by the user when calling setNativeDataType
            // no data has been written yet
            std::map<std::string, pylidar::SFieldInfo>::iterator itr = self->pAttributeTypeMap->find(pszField);
            if( itr != self->pAttributeTypeMap->end() )
            {
                /* Now build dtype string - easier than having a switch on all the combinations */
                char cKind = itr->second.cKind;
                int nSize = itr->second.nSize;
#if PY_MAJOR_VERSION >= 3
                PyObject *pString = PyUnicode_FromFormat("%c%d", cKind, nSize);
#else
                PyObject *pString = PyString_FromFormat("%c%d", cKind, nSize);
#endif
                /* assume success */
                PyArray_DescrConverter(pString, &pDescr);
                Py_DECREF(pString);
            }
        }

        if( (pDescr == NULL ) && (self->pHeader != NULL ) && (self->pHeader->number_attributes > 0 ) )
        {
            // we have written stuff, check out the las info - will have fields
            // with the same data type as the data - ie they didn't call setNativeDataType
            for( I32 index = 0; index < self->pHeader->number_attributes; index++ )
            {
                LASattribute *pAttr = &self->pPoint->attributer->attributes[index];
                
                // not in our 'compulsory' fields - must be an attribute
                if( strcmp(pAttr->name, pszField) == 0)
                {
                    // there is a method to do this, but bizarrely made private
                    I32 type = ((I32)pAttr->data_type - 1)%10;
                    int numpytype = 0;
                    switch(type)
                    {
                        case LAS_ATTRIBUTE_U8:
                            numpytype = NPY_UINT8;
                            break;
                        case LAS_ATTRIBUTE_I8:
                            numpytype = NPY_INT8;
                            break;
                        case LAS_ATTRIBUTE_U16:
                            numpytype = NPY_UINT16;
                            break;
                        case LAS_ATTRIBUTE_I16:
                            numpytype = NPY_INT16;
                            break;
                        case LAS_ATTRIBUTE_U32:
                            numpytype = NPY_UINT32;
                            break;
                        case LAS_ATTRIBUTE_I32:
                            numpytype = NPY_INT32;
                            break;
                        case LAS_ATTRIBUTE_U64:
                            numpytype = NPY_UINT64;
                            break;
                        case LAS_ATTRIBUTE_I64:
                            numpytype = NPY_INT64;
                            break;
                        case LAS_ATTRIBUTE_F32:
                            numpytype = NPY_FLOAT32;
                            break;
                        case LAS_ATTRIBUTE_F64:
                            numpytype = NPY_FLOAT64;
                            break;                
                    }

                    pDescr = PyArray_DescrFromType(numpytype);
                    break;
                }
                index++;
            }
        }
    }
    if( pDescr == NULL )
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC->error, "Unable to find data type for %s", pszField);
        return NULL;
    }

    return (PyObject*)pDescr;
}

static PyObject *PyLasFileWrite_setNativeDataType(PyLasFileWrite *self, PyObject *args)
{
    const char *pszField;
    PyObject *pPythonType;
    if( !PyArg_ParseTuple(args, "sO:setNativeDataType", &pszField, &pPythonType) )
        return NULL;

    if( !PyType_Check(pPythonType) )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Last argument needs to be python type");
        return NULL;
    }

    // now convert to a numpy dtype so we can get kind/size info
    PyArray_Descr *pDtype = NULL;
    if( !PyArray_DescrConverter(pPythonType, &pDtype) )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Could not convert python type to numpy dtype");
        return NULL;
    }

    std::set<std::string> nonAttrPointSet = getEssentialPointFieldNames();
    if( nonAttrPointSet.find(pszField) != nonAttrPointSet.end() )
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC->error, "Can't set data type for %s", pszField);
        return NULL;
    }

    pylidar::SFieldInfo info;
    info.cKind = pDtype->kind;
    info.nOffset = 0; // don't know yet
    info.nSize = pDtype->elsize;
    self->pAttributeTypeMap->insert(std::pair<std::string, pylidar::SFieldInfo>(pszField, info));

    // I *think* this is correct since I don't pass it to any of the (ref stealing)
    // array creation routines
    Py_DECREF(pDtype);

    Py_RETURN_NONE;
}

static PyObject *PyLasFileWrite_getScalingColumns(PyLasFileWrite *self, PyObject *args)
{
    PyObject *pString;
    PyObject *pColumns = PyList_New(0);
#if PY_MAJOR_VERSION >= 3
    pString = PyUnicode_FromString("X");
#else
    pString = PyString_FromString("X");
#endif
    PyList_Append(pColumns, pString);    
    Py_DECREF(pString);

#if PY_MAJOR_VERSION >= 3
    pString = PyUnicode_FromString("Y");
#else
    pString = PyString_FromString("Y");
#endif
    PyList_Append(pColumns, pString);    
    Py_DECREF(pString);

#if PY_MAJOR_VERSION >= 3
    pString = PyUnicode_FromString("Z");
#else
    pString = PyString_FromString("Z");
#endif
    PyList_Append(pColumns, pString);    
    Py_DECREF(pString);

    if( self->pScalingMap != NULL )
    {
        for(std::map<std::string, std::pair<double, double> >::iterator itr = self->pScalingMap->begin();
                itr != self->pScalingMap->end(); itr++ )
        {
#if PY_MAJOR_VERSION >= 3
            pString = PyUnicode_FromString(itr->first.c_str());
#else
            pString = PyString_FromString(itr->first.c_str());
#endif
            PyList_Append(pColumns, pString);    
            Py_DECREF(pString);
        }
    }
    return pColumns;
}

/* Table of methods */
static PyMethodDef PyLasFileWrite_methods[] = {
    {"writeData", (PyCFunction)PyLasFileWrite_writeData, METH_VARARGS, NULL}, 
    {"setEPSG", (PyCFunction)PyLasFileWrite_setEPSG, METH_VARARGS, NULL},
    {"setScaling", (PyCFunction)PyLasFileWrite_setScaling, METH_VARARGS, NULL},
    {"getNativeDataType", (PyCFunction)PyLasFileWrite_getNativeDataType, METH_VARARGS, NULL},
    {"setNativeDataType", (PyCFunction)PyLasFileWrite_setNativeDataType, METH_VARARGS, NULL},
    {"getScalingColumns", (PyCFunction)PyLasFileWrite_getScalingColumns, METH_NOARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLasFileWriteType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_las.FileWrite",         /*tp_name*/
    sizeof(PyLasFileWrite),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyLasFileWrite_dealloc, /*tp_dealloc*/
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
    "Las File Writer object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyLasFileWrite_methods,             /* tp_methods */
    0,             /* tp_members */
    0,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyLasFileWrite_init,      /* tp_init */
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
    PyModule_AddObject(pModule, "error", state->error);

    /* las file read type */
    PyLasFileReadType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyLasFileReadType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyLasFileReadType);
    PyModule_AddObject(pModule, "LasFileRead", (PyObject *)&PyLasFileReadType);

    /* las file write type */
    PyLasFileWriteType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyLasFileWriteType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyLasFileWriteType);
    PyModule_AddObject(pModule, "LasFileWrite", (PyObject *)&PyLasFileWriteType);

    // module constants
    PyModule_AddIntConstant(pModule, "FIRST_RETURN", FIRST_RETURN);
    PyModule_AddIntConstant(pModule, "LAST_RETURN", LAST_RETURN);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
