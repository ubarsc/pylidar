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

#include <cmath>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "pylvector.h"

#include "lasreader.hpp"

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
    npy_uint32 pts_start_idx;
    npy_uint8 number_of_returns;
    npy_uint8 orig_number_of_returns; // original in file. != number_of_returns when BUILD_PULSES=False
    double gps_time;
    npy_uint8 scan_direction_flag;
    npy_uint8 edge_of_flight_line;
    double x_origin; // the following only set when we have waveforms, or multiple returns
    double y_origin;
    double z_origin;
    double azimuth;
    double zenith;
    double x_idx;
    double y_idx;
} SLasPulse;

/* field info for pylidar_structArrayToNumpy */
static SpylidarFieldDefn LasPulseFields[] = {
    CREATE_FIELD_DEFN(SLasPulse, scan_angle_rank, 'i'),
    CREATE_FIELD_DEFN(SLasPulse, number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, pts_start_idx, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, orig_number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, gps_time, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, scan_direction_flag, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, edge_of_flight_line, 'u'),
    CREATE_FIELD_DEFN(SLasPulse, x_origin, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, y_origin, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, z_origin, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, azimuth, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, zenith, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, x_idx, 'f'),
    CREATE_FIELD_DEFN(SLasPulse, y_idx, 'f'),
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
} PyLasFile;

static const char *SupportedDriverOptions[] = {"BUILD_PULSES", "BIN_SIZE", "PULSE_INDEX", NULL};
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
    if(self->pWaveformReader != NULL)
    {
        delete self->pWaveformReader;
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
    self->nPulsesRead = 0;
    self->fBinSize = 0;

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
    PyDict_SetItemString(pHeaderDict, "MAX_X", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_x);
    PyDict_SetItemString(pHeaderDict, "MIN_X", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_y);
    PyDict_SetItemString(pHeaderDict, "MAX_Y", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_y);
    PyDict_SetItemString(pHeaderDict, "MIN_Y", pVal);

    pVal = PyFloat_FromDouble(pHeader->max_z);
    PyDict_SetItemString(pHeaderDict, "MAX_Z", pVal);

    pVal = PyFloat_FromDouble(pHeader->min_z);
    PyDict_SetItemString(pHeaderDict, "MIN_Z", pVal);

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
static PyObject *PyLasFile_readData(PyLasFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
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
    pylidar::CVector<SLasPoint> points(nInitSize, nGrowBy);
    pylidar::CVector<SLasWaveformInfo> waveformInfos(nInitSize, nGrowBy);
    pylidar::CVector<U8> received(nInitSize, nGrowBy);
    SLasPulse lasPulse;
    SLasPoint lasPoint;
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

        // always add a new point
        lasPoint.x = self->pReader->get_x();
        lasPoint.y = self->pReader->get_y();
        lasPoint.z = self->pReader->get_z();
        lasPoint.intensity = pPoint->intensity;
        lasPoint.return_number = pPoint->return_number - 1; // 1-based for some reason
        lasPoint.classification = pPoint->classification;
        lasPoint.synthetic_flag = pPoint->synthetic_flag;
        lasPoint.keypoint_flag = pPoint->keypoint_flag;
        lasPoint.withheld_flag = pPoint->withheld_flag;
        lasPoint.user_data = pPoint->user_data;
        lasPoint.point_source_ID = pPoint->point_source_ID;
        lasPoint.red = pPoint->rgb[0];        
        lasPoint.green = pPoint->rgb[1];        
        lasPoint.blue = pPoint->rgb[2];        
        lasPoint.nir = pPoint->rgb[3];        
        points.push(&lasPoint);

        // only add a pulse if we are building a pulse per point (self->bBuildPulses == false)
        // or this is the first return of a number of points
        if( !self->bBuildPulses || ( lasPoint.return_number == 0 ) )
        {
            lasPulse.scan_angle_rank = pPoint->scan_angle_rank;
            lasPulse.pts_start_idx = points.getNumElems();
            if( self->bBuildPulses )
                lasPulse.number_of_returns = pPoint->number_of_returns;
            else
                lasPulse.number_of_returns = 1;

            lasPulse.orig_number_of_returns = pPoint->number_of_returns;
            lasPulse.gps_time = pPoint->gps_time;
            lasPulse.scan_direction_flag = pPoint->scan_direction_flag;
            lasPulse.edge_of_flight_line = pPoint->edge_of_flight_line;

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
                lasPulse.x_origin = lasPoint.x - location * self->pWaveformReader->XYZt[0];
                lasPulse.y_origin = lasPoint.y - location * self->pWaveformReader->XYZt[1];
                lasPulse.z_origin = lasPoint.z - location * self->pWaveformReader->XYZt[2];

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

    // go through all pulses and find those with 
    // number_of_returns > 1 and use point locations to fill in 
    // zenith, azimuth etc
    if( self->bBuildPulses )
    {
        for( npy_intp nPulseCount = 0; nPulseCount < pulses.getNumElems(); nPulseCount++)
        {
            SLasPulse *pPulse = pulses.getElem(nPulseCount);
            if( ( pPulse->number_of_returns > 1 ) && ( pPulse->zenith == 0 ) && ( pPulse->azimuth == 0) )
            {
                SLasPoint *p1 = points.getElem(pPulse->pts_start_idx);
                SLasPoint *p2 = points.getElem(pPulse->pts_start_idx + pPulse->number_of_returns);
                ConvertCoordsToAngles(p2->x, p1->x, p2->y, p1->y, p2->z, p1->z,
                            &pPulse->zenith, &pPulse->azimuth);
            }
        }
    }

    PyObject *pPulses = pulses.getNumpyArray(LasPulseFields);
    PyObject *pPoints = points.getNumpyArray(LasPointFields);
    PyObject *pInfos = waveformInfos.getNumpyArray(LasWaveformInfoFields);
    PyObject *pReceived = received.getNumpyArray(NPY_UINT8);

    // build tuple
    PyObject *pTuple = PyTuple_Pack(4, pPulses, pPoints, pInfos, pReceived);
    return pTuple;
}

static PyObject *PyLasFile_getEPSG(PyLasFile *self, PyObject *args)
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
    for (int j = 0; j < pHeader->vlr_geo_keys->number_of_keys; j++)
    {
        if(pHeader->vlr_geo_key_entries[j].key_id == 3072)
        {
            nEPSG = pHeader->vlr_geo_key_entries[j].value_offset;
            foundProjection = true;
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

static PyObject *PyLasFile_setExtent(PyLasFile *self, PyObject *args)
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

/* Table of methods */
static PyMethodDef PyLasFile_methods[] = {
    {"readHeader", (PyCFunction)PyLasFile_readHeader, METH_NOARGS, NULL},
    {"readData", (PyCFunction)PyLasFile_readData, METH_VARARGS, NULL}, 
    {"getEPSG", (PyCFunction)PyLasFile_getEPSG, METH_NOARGS, NULL},
    {"setExtent", (PyCFunction)PyLasFile_setExtent, METH_VARARGS, NULL},
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

static PyObject *PyLasFile_getPulsesRead(PyLasFile *self, void *closure)
{
    return PyLong_FromSsize_t(self->nPulsesRead);
}

static PyObject *PyLasFile_getBinSize(PyLasFile *self, void *closure)
{
    return PyFloat_FromDouble(self->fBinSize);
}

static int PyLasFile_setBinSize(PyLasFile *self, PyObject *value, void *closure)
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
static PyGetSetDef PyLasFile_getseters[] = {
    {"build_pulses", (getter)PyLasFile_getBuildPulses, NULL, 
        "Whether we are building pulses of multiple points when reading", NULL},
    {"hasSpatialIndex", (getter)PyLasFile_getHasSpatialIndex, NULL,
        "Whether a spatial index exists for this file", NULL},
    {"finished", (getter)PyLasFile_getFinished, NULL, 
        "Whether we have finished reading the file or not", NULL},
    {"pulsesRead", (getter)PyLasFile_getPulsesRead, NULL,
        "Number of pulses read", NULL},
    {"binSize", (getter)PyLasFile_getBinSize, (setter)PyLasFile_setBinSize,
        "Bin size to use for spatial data", NULL},
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

    // module constants
    PyModule_AddIntConstant(pModule, "FIRST_RETURN", FIRST_RETURN);
    PyModule_AddIntConstant(pModule, "LAST_RETURN", LAST_RETURN);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
