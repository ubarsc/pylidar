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

#include "lasreader.hpp"
#include "laswriter.hpp"

// for CVector
static const int nGrowBy = 1000;
static const int nInitSize = 40000;
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
#else
#define GETSTATE(m) (&_state)
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
} SLasPulse;

/* field info for pylidar_structArrayToNumpy */
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

/* field info for pylidar_structArrayToNumpy */
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

/* field info for pylidar_structArrayToNumpy */
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
} PyLasFileRead;

static const char *SupportedDriverOptionsRead[] = {"BUILD_PULSES", "BIN_SIZE", "PULSE_INDEX", NULL};
static PyObject *las_getReadSupportedOptions(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(SupportedDriverOptionsRead);
}

static const char *SupportedDriverOptionsWrite[] = {"FORMAT", "RECORD_LENGTH", NULL};
static PyObject *las_getWriteSupportedOptions(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(SupportedDriverOptionsWrite);
}

// module methods
static PyMethodDef module_methods[] = {
    {"getReadSupportedOptions", (PyCFunction)las_getReadSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options for reading"},
    {"getWriteSupportedOptions", (PyCFunction)las_getWriteSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options for writing"},
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
    self->nPulsesRead = 0;
    self->fBinSize = 0;
    self->nPulseIndex = FIRST_RETURN;
    self->pLasPointFieldsWithExt = NULL;

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
        if( PyLong_Check(pPulseIndex) )
        {
            self->nPulseIndex = PyLong_AsLong(pPulseIndex);
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
            PyErr_SetString(GETSTATE(m)->error, "PULSE_INDEX must be an int");    
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

    pVal = PyLong_FromLong(pHeader->file_source_ID);
    PyDict_SetItemString(pHeaderDict, "FILE_SOURCE_ID", pVal);

    pVal = PyLong_FromLong(pHeader->global_encoding);
    PyDict_SetItemString(pHeaderDict, "GLOBAL_ENCODING", pVal);

    pVal = PyLong_FromLong(pHeader->project_ID_GUID_data_1);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_1", pVal);

    pVal = PyLong_FromLong(pHeader->project_ID_GUID_data_2);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_2", pVal);

    pVal = PyLong_FromLong(pHeader->project_ID_GUID_data_3);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_3", pVal);

    pylidar::CVector<U8> project_ID_GUID_data_4Vector(pHeader->project_ID_GUID_data_4, 
                            sizeof(pHeader->project_ID_GUID_data_4));    
    pVal = project_ID_GUID_data_4Vector.getNumpyArray(NPY_UINT8);
    PyDict_SetItemString(pHeaderDict, "PROJECT_ID_GUID_DATA_4", pVal);

    pVal = PyLong_FromLong(pHeader->version_major);
    PyDict_SetItemString(pHeaderDict, "VERSION_MAJOR", pVal);

    pVal = PyLong_FromLong(pHeader->version_minor);
    PyDict_SetItemString(pHeaderDict, "VERSION_MINOR", pVal);

#if PY_MAJOR_VERSION >= 3
    pVal = PyUnicode_FromStringAndSize(pHeader->system_identifier, 
                GET_LENGTH(pHeader->system_identifier));
#else
    pVal = PyString_FromStringAndSize(pHeader->system_identifier, 
                GET_LENGTH(pHeader->system_identifier));
#endif
    PyDict_SetItemString(pHeaderDict, "SYSTEM_IDENTIFIER", pVal);

#if PY_MAJOR_VERSION >= 3
    pVal = PyUnicode_FromStringAndSize(pHeader->generating_software, 
                GET_LENGTH(pHeader->generating_software));
#else
    pVal = PyString_FromStringAndSize(pHeader->generating_software, 
                GET_LENGTH(pHeader->generating_software));
#endif
    PyDict_SetItemString(pHeaderDict, "GENERATING_SOFTWARE", pVal);

    pVal = PyLong_FromLong(pHeader->file_creation_day);
    PyDict_SetItemString(pHeaderDict, "FILE_CREATION_DAY", pVal);

    pVal = PyLong_FromLong(pHeader->file_creation_year);
    PyDict_SetItemString(pHeaderDict, "FILE_CREATION_YEAR", pVal);

    pVal = PyLong_FromLong(pHeader->header_size);
    PyDict_SetItemString(pHeaderDict, "HEADER_SIZE", pVal);

    pVal = PyLong_FromLong(pHeader->offset_to_point_data);
    PyDict_SetItemString(pHeaderDict, "OFFSET_TO_POINT_DATA", pVal);

    pVal = PyLong_FromLong(pHeader->number_of_variable_length_records);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_VARIABLE_LENGTH_RECORDS", pVal);

    pVal = PyLong_FromLong(pHeader->point_data_format);
    PyDict_SetItemString(pHeaderDict, "POINT_DATA_FORMAT", pVal);

    pVal = PyLong_FromLong(pHeader->point_data_record_length);
    PyDict_SetItemString(pHeaderDict, "POINT_DATA_RECORD_LENGTH", pVal);

    pVal = PyLong_FromLong(pHeader->number_of_point_records);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_POINT_RECORDS", pVal);

    pylidar::CVector<U32> number_of_points_by_returnVector(pHeader->number_of_points_by_return, 
                            sizeof(pHeader->number_of_points_by_return));    
    pVal = number_of_points_by_returnVector.getNumpyArray(NPY_UINT32);
    PyDict_SetItemString(pHeaderDict, "NUMBER_OF_POINTS_BY_RETURN", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_x);
    PyDict_SetItemString(pHeaderDict, "X_MAX", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_x);
    PyDict_SetItemString(pHeaderDict, "X_MIN", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_y);
    PyDict_SetItemString(pHeaderDict, "Y_MAX", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_y);
    PyDict_SetItemString(pHeaderDict, "Y_MIN", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_z);
    PyDict_SetItemString(pHeaderDict, "Z_MAX", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_z);
    PyDict_SetItemString(pHeaderDict, "Z_MIN", pVal);

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
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "readData either takes 2 params for non-spatial reads, or 0 params for spatial reads");
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

    // spatial reads go until there is no more data (see setExtent)
    // non-spatial reads get bFinished updated.
    while( !bFinished )
    {
        if( !self->pReader->read_point() )
        {
            self->bFinished = true;
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

            // copy the data over
            memcpy(self->pLasPointFieldsWithExt, LasPointFields, sizeof(SpylidarFieldDefn) * nOldFields);

            // assume extra fields are all floats which seems to make sense as even
            // if they are stored as short etc, there is scaling often applied.
            int nNewTotalSize = sizeof(SLasPoint) + 
                    (sizeof(double) * pPoint->attributer->number_attributes);

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
                pDest->nOffset = sizeof(SLasPoint) + (i * sizeof(double));
                // nStructTotalSize done above
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

        // only add a pulse if we are building a pulse per point (self->bBuildPulses == false)
        // or this is the first return of a number of points
        if( !self->bBuildPulses || ( pLasPoint->return_number == 0 ) )
        {
            lasPulse.scan_angle_rank = pPoint->get_scan_angle_rank();
            lasPulse.scan_angle = pPoint->get_scan_angle();
            lasPulse.pts_start_idx = pPoints->getNumElems();
            if( self->bBuildPulses )
                lasPulse.number_of_returns = pPoint->get_number_of_returns();
            else
                lasPulse.number_of_returns = 1;

            lasPulse.orig_number_of_returns = pPoint->get_number_of_returns();
            lasPulse.gps_time = pPoint->get_gps_time();
            lasPulse.scan_direction_flag = pPoint->get_scan_direction_flag();
            lasPulse.edge_of_flight_line = pPoint->get_edge_of_flight_line();
            lasPulse.scanner_channel = pPoint->get_extended_scanner_channel(); // 0 if not 'extended'

            if( self->pWaveformReader != NULL )
            {
                // we have waveforms
                self->pWaveformReader->read_waveform(pPoint);

                // fill in the info                
                lasWaveformInfo.number_of_waveform_received_bins = self->pWaveformReader->nsamples;
                lasWaveformInfo.received_start_idx = received.getNumElems();
                U8 lasindex = pPoint->wavepacket.getIndex();
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
            }

            pulses.push(&lasPulse);
        }

        // update loop exit for non-spatial reads
        // spatial reads keep going until all the way through the file
        if( PyTuple_Size(args) == 2 )
        {
            bFinished = (pulses.getNumElems() >= nPulses);
        }
    }

    self->nPulsesRead += pulses.getNumElems();

    // go through all the pulses and do some tidying up
    for( npy_intp nPulseCount = 0; nPulseCount < pulses.getNumElems(); nPulseCount++)
    {
        SLasPulse *pPulse = pulses.getElem(nPulseCount);
        SLasPoint *p1 = pPoints->getElem(pPulse->pts_start_idx);
        SLasPoint *p2 = pPoints->getElem(pPulse->pts_start_idx + pPulse->number_of_returns);

        // set x_idx and y_idx for the pulses
        if( pPulse->number_of_returns > 0 )
        {
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

    PyObject *pNumpyPulses = pulses.getNumpyArray(LasPulseFields);
    SpylidarFieldDefn *pPointDefn = LasPointFields;
    if( self->pLasPointFieldsWithExt != NULL )
    {
        // extra fields - use other definition instead
        pPointDefn = self->pLasPointFieldsWithExt;
    }

    PyObject *pNumpyPoints = pPoints->getNumpyArray(pPointDefn);
    delete pPoints;
    PyObject *pNumpyInfos = waveformInfos.getNumpyArray(LasWaveformInfoFields);
    PyObject *pNumpyReceived = received.getNumpyArray(NPY_UINT8);

    // build tuple
    PyObject *pTuple = PyTuple_Pack(4, pNumpyPulses, pNumpyPoints, pNumpyInfos, pNumpyReceived);
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
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, "Cannot find EPSG code for Coordinate System");
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

    double dGain, dOffset;
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
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_Format(GETSTATE(m)->error, "Unable to get scaling for field %s", pszField);
        return NULL;
    }

    return Py_BuildValue("dd", dGain, dOffset);
}

/* Table of methods */
static PyMethodDef PyLasFileRead_methods[] = {
    {"readHeader", (PyCFunction)PyLasFileRead_readHeader, METH_NOARGS, NULL},
    {"readData", (PyCFunction)PyLasFileRead_readData, METH_VARARGS, NULL}, 
    {"getEPSG", (PyCFunction)PyLasFileRead_getEPSG, METH_NOARGS, NULL},
    {"setExtent", (PyCFunction)PyLasFileRead_setExtent, METH_VARARGS, NULL},
    {"getScaling", (PyCFunction)PyLasFileRead_getScaling, METH_VARARGS, NULL},
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
    {"build_pulses", (getter)PyLasFileRead_getBuildPulses, NULL, 
        "Whether we are building pulses of multiple points when reading", NULL},
    {"hasSpatialIndex", (getter)PyLasFileRead_getHasSpatialIndex, NULL,
        "Whether a spatial index exists for this file", NULL},
    {"finished", (getter)PyLasFileRead_getFinished, NULL, 
        "Whether we have finished reading the file or not", NULL},
    {"pulsesRead", (getter)PyLasFileRead_getPulsesRead, NULL,
        "Number of pulses read", NULL},
    {"binSize", (getter)PyLasFileRead_getBinSize, (setter)PyLasFileRead_setBinSize,
        "Bin size to use for spatial data", NULL},
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

// to help us remember info for each field without having to look it up each time
// used by CFieldInfoMap
typedef struct {
    char cKind;
    int nOffset;
    int nSize;
} SFieldInfo;

/* Python object wrapping a LASwriter */
typedef struct {
    PyObject_HEAD
    char *pszFilename;
    LASwriter *pWriter;
    LASheader *pHeader;
    LASpoint *pPoint;
    int nEPSG;
    bool bXScalingSet;
    bool bYScalingSet;
    bool bZScalingSet;
    F64 dXGain;
    F64 dXOffset;
    F64 dYGain;
    F64 dYOffset;
    F64 dZGain;
    F64 dZOffset;
    U8 point_data_format;
    U16 point_data_record_length;
    std::vector<SFieldInfo> *pAttributeFields; // index is the position in the vector
    std::map<std::string, std::pair<double, double> > *pScalingMap;
} PyLasFileWrite;


/* destructor - close and delete */
static void 
PyLasFileWrite_dealloc(PyLasFileWrite *self)
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
    if(self->pPoint != NULL)
    {
        delete self->pPoint;
    }
    if(self->pAttributeFields != NULL)
    {
        delete self->pAttributeFields;
    }
    if(self->pScalingMap != NULL)
    {
        delete self->pScalingMap;
    }
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
    if( pOptionDict != Py_None )
    {
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

        PyObject *pVal = PyDict_GetItemString(pOptionDict, "FORMAT");
        if( pVal != NULL )
        {
            if( !PyLong_Check(pVal) )
            {
                // raise Python exception
                PyObject *m;
#if PY_MAJOR_VERSION >= 3
                // best way I could find for obtaining module reference
                // from inside a class method. Not needed for Python < 3.
                m = PyState_FindModule(&moduledef);
#endif
                PyErr_SetString(GETSTATE(m)->error, "FORMAT parameter must be integer");
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
                PyObject *m;
#if PY_MAJOR_VERSION >= 3
                // best way I could find for obtaining module reference
                // from inside a class method. Not needed for Python < 3.
                m = PyState_FindModule(&moduledef);
#endif
                PyErr_SetString(GETSTATE(m)->error, "RECORD_LENGTH parameter must be integer");
                return -1;
            }
            self->point_data_record_length = PyLong_AsLong(pVal);
        }
    }

    // set up when first block written
    // so user can update the header
    self->pWriter = NULL;
    self->pHeader = NULL;
    self->pPoint = NULL;
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
    self->pAttributeFields = NULL;
    self->pScalingMap = new std::map<std::string, std::pair<double, double> >;

    // copy filename so we can open later
    self->pszFilename = strdup(pszFname);

    return 0;
}

#define DO_INT64_READ(tempVar) memcpy(&tempVar, (char*)pRow + info.nOffset, sizeof(tempVar)); \
            nRetVal = (npy_int64)tempVar; 

#define DO_FLOAT64_READ(tempVar) memcpy(&tempVar, (char*)pRow + info.nOffset, sizeof(tempVar)); \
            dRetVal = (double)tempVar; 

class CFieldInfoMap : public std::map<std::string, SFieldInfo>
{
public:
    CFieldInfoMap(PyObject *pArray) 
    {
        SFieldInfo info;
        PyArray_Descr *pDescr = PyArray_DESCR(pArray);
        PyObject *pKeys = PyDict_Keys(pDescr->fields);
        for( Py_ssize_t i = 0; i < PyList_Size(pKeys); i++)
        {
            PyObject *pKey = PyList_GetItem(pKeys, i);
#if PY_MAJOR_VERSION >= 3
            PyObject *bytesKey = PyUnicode_AsEncodedString(pKey, NULL, NULL);
            char *pszElementName = PyBytes_AsString(bytesKey);
#else
            char *pszElementName = PyString_AsString(pKey);
#endif

            pylidar_getFieldDescr(pArray, pszElementName, &info.nOffset, &info.cKind, &info.nSize, NULL);
            insert( std::pair<std::string, SFieldInfo>(pszElementName, info) );

#if PY_MAJOR_VERSION >= 3
            Py_DECREF(bytesKey);
#endif
        }
        Py_DECREF(pKeys);
    }

    npy_int64 getIntValue(std::string sName, void *pRow)
    {
        npy_char nCharVal;
        npy_bool nBoolVal;
        npy_byte nByteVal;
        npy_ubyte nUByteVal;
        npy_short nShortVal;
        npy_ushort nUShortVal;
        npy_int nIntVal;
        npy_uint nUIntVal;
        npy_long nLongVal;
        npy_ulong nULongVal;
        npy_float fFloatVal;
        npy_double fDoubleVal;
        npy_int64 nRetVal=0;

        iterator it = find(sName);
        if( it == end() )
        {
            return 0;
        }
        SFieldInfo info = it->second;
        if( ( info.cKind == 'b' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nBoolVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nByteVal);
        }
        else if ( ( info.cKind == 'S' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nCharVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 1 ) )
        {
            DO_INT64_READ(nUByteVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 2 ) )
        {
            DO_INT64_READ(nShortVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 2 ) )
        {
            DO_INT64_READ(nUShortVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 4 ) )
        {
            DO_INT64_READ(nIntVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 4 ) )
        {
            DO_INT64_READ(nUIntVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 8 ) )
        {
            DO_INT64_READ(nLongVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 8 ) )
        {
            DO_INT64_READ(nULongVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 4 ) )
        {
            DO_INT64_READ(fFloatVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 8 ) )
        {
            DO_INT64_READ(fDoubleVal);
        }
        return nRetVal;        
    }
    double getDoubleValue(std::string sName, void *pRow)
    {
        npy_char nCharVal;
        npy_bool nBoolVal;
        npy_byte nByteVal;
        npy_ubyte nUByteVal;
        npy_short nShortVal;
        npy_ushort nUShortVal;
        npy_int nIntVal;
        npy_uint nUIntVal;
        npy_long nLongVal;
        npy_ulong nULongVal;
        npy_float fFloatVal;
        npy_double fDoubleVal;
        double dRetVal=0;

        iterator it = find(sName);
        if( it == end() )
        {
            return 0;
        }

        SFieldInfo info = it->second;
        if( ( info.cKind == 'b' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nBoolVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nByteVal);
        }
        else if ( ( info.cKind == 'S' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nCharVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 1 ) )
        {
            DO_FLOAT64_READ(nUByteVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 2 ) )
        {
            DO_FLOAT64_READ(nShortVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 2 ) )
        {
            DO_FLOAT64_READ(nUShortVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 4 ) )
        {
            DO_FLOAT64_READ(nIntVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 4 ) )
        {
            DO_FLOAT64_READ(nUIntVal);
        }
        else if ( ( info.cKind == 'i' ) && ( info.nSize == 8 ) )
        {
            DO_FLOAT64_READ(nLongVal);
        }
        else if ( ( info.cKind == 'u' ) && ( info.nSize == 8 ) )
        {
            DO_FLOAT64_READ(nULongVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 4 ) )
        {
            DO_FLOAT64_READ(fFloatVal);
        }
        else if ( ( info.cKind == 'f' ) && ( info.nSize == 8 ) )
        {
            DO_FLOAT64_READ(fDoubleVal);
        }
        return dRetVal;        
    }

};

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
        for( npy_intp i = 0; i < PyArray_DIM(pArray, 0); i++ )
        {
            pHeader->project_ID_GUID_data_4[i] = *((U8*)PyArray_GETPTR1(pArray, i));
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

    pVal = PyDict_GetItemString(pHeaderDict, "POINT_DATA_FORMAT");
    if( pVal != NULL )
        pHeader->point_data_format = PyLong_AsLong(pVal);

    pVal = PyDict_GetItemString(pHeaderDict, "POINT_DATA_RECORD_LENGTH");
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
        for( npy_intp i = 0; i < PyArray_DIM(pArray, 0); i++ )
        {
            pHeader->number_of_points_by_return[i] = *((U8*)PyArray_GETPTR1(pArray, i));
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

// returns a set of names we recognise and match to 
// 'essential' fields in the LASpoint structure
std::set<std::string> getEssentialPointFieldNames()
{
    std::set<std::string> nonAttrPointSet;
    nonAttrPointSet.insert("EXTENDED_POINT_TYPE");
    nonAttrPointSet.insert("X");
    nonAttrPointSet.insert("Y");
    nonAttrPointSet.insert("Z");
    nonAttrPointSet.insert("INTENSITY");
    nonAttrPointSet.insert("CLASSIFICATION");
    nonAttrPointSet.insert("SYNTHETIC_FLAG");
    nonAttrPointSet.insert("KEYPOINT_FLAG");
    nonAttrPointSet.insert("WITHHELD_FLAG");
    nonAttrPointSet.insert("USER_DATA");
    nonAttrPointSet.insert("POINT_SOURCE_ID");
    nonAttrPointSet.insert("DELETED_FLAG");
    nonAttrPointSet.insert("RED");
    nonAttrPointSet.insert("GREEN");
    nonAttrPointSet.insert("BLUE");
    nonAttrPointSet.insert("NIR");
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
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_SetString(GETSTATE(m)->error, "First parameter to writeData must be header dictionary");
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
        pszMessage = "Waveform info must be a numpy array";
    }
    if( bArraysOk && ((PyArray_NDIM(pPulses) != 1) || (PyArray_NDIM(pPoints) != 2) || 
            (bHaveWaveformInfos && (PyArray_NDIM(pWaveformInfos) != 2)) || 
            (bHaveReceived && (PyArray_NDIM(pReceived) != 3)) ) )
    {
        bArraysOk = false;
        pszMessage = "pulses must be 1d, points and received 2d and wavforminfo 3d";
    }


    if( !bArraysOk )
    {
        // raise Python exception
        PyObject *m;
#if PY_MAJOR_VERSION >= 3
        // best way I could find for obtaining module reference
        // from inside a class method. Not needed for Python < 3.
        m = PyState_FindModule(&moduledef);
#endif
        PyErr_SetString(GETSTATE(m)->error, pszMessage);
        return NULL;
    }

    // create a mapping between names of fields and their 'info' which has
    // the type, offset etc
    // do it up here in case we need info for the attributes when setting
    // up the header and writer
    CFieldInfoMap pulseMap(pPulses);
    CFieldInfoMap pointMap(pPoints);

    if( self->pWriter == NULL )
    {
        // check that all the scaling has been set
        if( !self->bXScalingSet || !self->bYScalingSet || !self->bZScalingSet )
        {
            // raise Python exception
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_SetString(GETSTATE(m)->error, "Must set scaling for X, Y and Z columns before writing data");
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
        self->pAttributeFields = new std::vector<SFieldInfo>();
        // temp set of the fields we use to populate LASpoint fields
        // others must be attributes
        std::set<std::string> nonAttrPointSet = getEssentialPointFieldNames();

        // iterate over our field map and see what isn't in nonAttrPointSet
        for( std::map<std::string, SFieldInfo>::iterator itr = pointMap.begin(); itr != pointMap.end(); itr++ )
        {
            if( nonAttrPointSet.find(itr->first) == nonAttrPointSet.end() )
            {
                // not in our 'compulsory' fields - must be an attribute
                self->pAttributeFields->push_back(itr->second);

                // tell the header
                U32 type = 0;
                char cKind = itr->second.cKind;
                int nSize = itr->second.nSize;
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

                self->pHeader->add_attribute(attr);
            }
        }

        // add a vlr if we actually have attributes
        // note - I don't completely understand how this all works.
        if( !self->pAttributeFields->empty() )
        {
            self->pHeader->add_vlr("LASF_Spec\0\0\0\0\0\0", 4, self->pHeader->number_attributes*sizeof(LASattribute),
                    (U8*)self->pHeader->attributes, FALSE, "by Pylidar");
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
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_SetString(GETSTATE(m)->error, "Unable to open las file");
            return NULL;
        }
    }

    for( npy_intp nPulseIdx = 0; nPulseIdx < PyArray_DIM(pPulses, 0); nPulseIdx++)
    {
        void *pPulseRow = PyArray_GETPTR1(pPulses, nPulseIdx);
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
            // IF ADDING OTHER FIELDS also add to getEssentialPointFieldNames() above
            void *pPointRow = PyArray_GETPTR2(pPoints, nPointCount, nPulseIdx);
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
            I32 index = 0;
            for( std::vector<SFieldInfo>::iterator itr = self->pAttributeFields->begin();
                    itr != self->pAttributeFields->end(); itr++ )
            {
                self->pPoint->set_attribute(index, (U8*)pPointRow + itr->nOffset);
                index++;
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
            PyObject *m;
#if PY_MAJOR_VERSION >= 3
            // best way I could find for obtaining module reference
            // from inside a class method. Not needed for Python < 3.
            m = PyState_FindModule(&moduledef);
#endif
            PyErr_Format(GETSTATE(m)->error, "Unable to set scaling for field %s", pszField);
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

/* Table of methods */
static PyMethodDef PyLasFileWrite_methods[] = {
    {"writeData", (PyCFunction)PyLasFileWrite_writeData, METH_VARARGS, NULL}, 
    {"setEPSG", (PyCFunction)PyLasFileWrite_setEPSG, METH_VARARGS, NULL},
    {"setScaling", (PyCFunction)PyLasFileWrite_setScaling, METH_VARARGS, NULL},
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
