/*
 * riegl_rdb.cpp
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
#include <cmath>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "pylvector.h"

#include "riegl/rdb.h"
#include "riegl/rdb/default/attributes.h"

/* An exception object for this module */
/* created in the init function */
struct RieglRDBState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct RieglRDBState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
static struct RieglRDBState _state;
#endif

static const int nGrowBy = 10000;
static const int nInitSize = 256*256;
// number of records to read in from librdb at one time
static const int nTempSize = 100;

/* Structure for pulses */
typedef struct {
    npy_uint64 pulse_ID;
    npy_uint64 timestamp;
    float azimuth;
    float zenith;
    npy_uint32 scanline;
    npy_uint16 scanline_Idx;
    double x_Idx;
    double y_Idx;
    npy_uint32 pts_start_idx;
    npy_uint8 number_of_returns;
} SRieglRDBPulse;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPulseFields[] = {
    CREATE_FIELD_DEFN(SRieglRDBPulse, pulse_ID, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, timestamp, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, azimuth, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, zenith, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, scanline, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, scanline_Idx, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, y_Idx, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, x_Idx, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, pts_start_idx, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPulse, number_of_returns, 'u'),
    {NULL} // Sentinel
};

/* Structure for points */
typedef struct {
    npy_uint64 return_Number;
    npy_uint64 timestamp;
    float deviation_Return;
    npy_uint8 classification;
    double range;
    double rho_app;
    double amplitude_Return;
    double x;
    double y;
    double z;
} SRieglRDBPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPointFields[] = {
    CREATE_FIELD_DEFN(SRieglRDBPoint, return_Number, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, timestamp, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, deviation_Return, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, classification, 'u'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, range, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, rho_app, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, amplitude_Return, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, x, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, y, 'f'),
    CREATE_FIELD_DEFN(SRieglRDBPoint, z, 'f'),
    {NULL} // Sentinel
};

// buffer that we read the data in before copying to point/pulse structures
// NOTE: any changes must be reflected in RiegRDBReader::resetBuffers()
typedef struct
{
    // points
    npy_uint64 id; // riegl.id
    npy_uint64 timestamp; // riegl.timestamp
    npy_int32 deviation; // riegl.deviation
    npy_uint16 classification; // riegl.class
    double reflectance; // riegl.reflectance
    double amplitude; // riegl.amplitude
    double xyz[3]; // riegl.xyz
    
    // pulses
    npy_uint32 row;  // riegl.row
    npy_uint16 column; // riegl.column
    
    // info for attributing points to pulses
    npy_uint8 target_index; // riegl.target_index
    npy_uint8 target_count; // riegl.target_count

} RieglRDBBuffer;

// Returns false and sets Python exception if error code set
static bool CheckRDBResult(RDBResult code, RDBContext *pContext);

// For use in PyRieglRDBFile_* functions, return ret on error
#define CHECKRESULT_FILE(code, ret) if( !CheckRDBResult(code, self->pContext) ) return ret;

// For use in RiegRDBReader, return ret on error
#define CHECKRESULT_READER(code, ret) if( !CheckRDBResult(code, m_pContext) ) return ret;
// for automating the binding process
#define CHECKBIND_READER(attribute, dataType, buffer) if( !CheckRDBResult( \
                    rdb_pointcloud_query_select_bind( \
                        m_pContext, m_pQuerySelect, attribute, dataType, buffer, \
                        sizeof(RieglRDBBuffer)), m_pContext) ) return false;

typedef struct
{
    PyObject_HEAD
    char *pszFilename; // so we can re-create the file obj if needed
    RDBContext *pContext;
    RDBPointcloud *pPointCloud;
    RDBPointcloudQuerySelect *pQuerySelect; // != NULL if we currently have one going
    Py_ssize_t nCurrentPulse;  // if pQuerySelect != NULL this is the pulse number we are up to
    bool bFinishedReading;
    PyObject *pHeader;  // dictionary with header info

} PyRieglRDBFile;

// module methods
static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static int rieglrdb_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int rieglrdb_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_rieglrdb",
        NULL,
        sizeof(struct RieglRDBState),
        module_methods,
        NULL,
        rieglrdb_traverse,
        rieglrdb_clear,
        NULL
};
#endif

class RiegRDBReader
{
public:
    RiegRDBReader(RDBContext *pContext, RDBPointcloudQuerySelect *pQuerySelect)
    {
        m_pContext = pContext;
        m_pQuerySelect = pQuerySelect;
        m_nPulsesToIgnore = 0;
    }
    
    void setPulsesToIgnore(Py_ssize_t nPulsesToIgnore)
    {
        m_nPulsesToIgnore = nPulsesToIgnore;
    }
    
    // reset librdb to read into the start of our m_TempBuffer
    bool resetBuffers()
    {
        // NOTE: this must match definition of RieglRDBBuffer
        // I'm using the info from attribues.h rather than strings
        //  - not sure what is best practice
        // point
        CHECKBIND_READER(RDB_RIEGL_ID.name, RDBDataTypeUINT64, &m_TempBuffer.id)
        CHECKBIND_READER(RDB_RIEGL_TIMESTAMP.name, RDBDataTypeUINT64, &m_TempBuffer.timestamp)
        CHECKBIND_READER(RDB_RIEGL_DEVIATION.name, RDBDataTypeINT32, &m_TempBuffer.deviation)
        CHECKBIND_READER(RDB_RIEGL_CLASS.name, RDBDataTypeUINT16, &m_TempBuffer.classification)
        CHECKBIND_READER(RDB_RIEGL_REFLECTANCE.name, RDBDataTypeDOUBLE, &m_TempBuffer.reflectance)
        CHECKBIND_READER(RDB_RIEGL_AMPLITUDE.name, RDBDataTypeDOUBLE, &m_TempBuffer.amplitude)
        CHECKBIND_READER(RDB_RIEGL_XYZ.name, RDBDataTypeDOUBLE, &m_TempBuffer.xyz)
        
        // these 2 don't appear to be documented, but are in there
        CHECKBIND_READER("riegl.row", RDBDataTypeUINT32, &m_TempBuffer.row);
        CHECKBIND_READER("riegl.column", RDBDataTypeUINT16, &m_TempBuffer.column);
        
        CHECKBIND_READER(RDB_RIEGL_TARGET_INDEX.name, RDBDataTypeUINT8, &m_TempBuffer.target_index)
        CHECKBIND_READER(RDB_RIEGL_TARGET_COUNT.name, RDBDataTypeUINT8, &m_TempBuffer.target_count)
        
        return true;
    }
    
    PyObject* readData(Py_ssize_t nPulses, Py_ssize_t &nCurrentPulse, bool &bFinished)
    {
        pylidar::CVector<SRieglRDBPulse> pulses(nInitSize, nGrowBy);
        SRieglRDBPulse pulse;
        pylidar::CVector<SRieglRDBPoint> points(nInitSize, nGrowBy);
        SRieglRDBPoint point;
        
        while( m_nPulsesToIgnore > 0 )
        {
            if( !resetBuffers() )
            {
                return NULL;
            }

            uint32_t processed = 0;
            if( rdb_pointcloud_query_select_next(
                    m_pContext, m_pQuerySelect, 1,
                    &processed) == 0 )
            {
                // 0 means 'end of file' apparently
                PyErr_SetString(GETSTATE_FC->error, "Unable to ignore correct number of pulses");
                return NULL;
            }
            m_nPulsesToIgnore--;
        }

        while( true )
        {
            // because we are reading in a number of pulses
            // and we can't rewind, we have to read in one point
            // at a time and slowly build up to the right number 
            // of pulses
        
            // reset to beginning of our temp space
            // have to do this every time since we are just reading one
            // - could be slow.
            if( !resetBuffers() )
            {
                return NULL;
            }

            uint32_t processed = 0;
            if( rdb_pointcloud_query_select_next(
                    m_pContext, m_pQuerySelect, 1,
                    &processed) == 0 )
            {
                // 0 means 'end of file' apparently
                bFinished = true;
                break;
            }
                    
            if( processed == 0 ) 
            {
                bFinished = true;
                break;
            }
            
            point.return_Number = m_TempBuffer.id;
            point.timestamp = m_TempBuffer.timestamp;
            point.deviation_Return = m_TempBuffer.deviation;
            point.classification = m_TempBuffer.classification;
            point.range = std::sqrt(std::pow(m_TempBuffer.xyz[0], 2) + 
                                    std::pow(m_TempBuffer.xyz[1], 2) + 
                                    std::pow(m_TempBuffer.xyz[2], 2));
            // Rescale reflectance from dB to papp
            point.rho_app = std::pow(10.0, m_TempBuffer.reflectance / 10.0);
            point.amplitude_Return = m_TempBuffer.amplitude;
            point.x = m_TempBuffer.xyz[0];
            point.y = m_TempBuffer.xyz[1];
            point.z = m_TempBuffer.xyz[2];
            
            if( m_TempBuffer.target_index == 1 ) 
            {
                // start of new pulse 
                  
                // check that we have the required number of pulses
                // (would have just finished reading all the attached points
                // of the previous pulse)
                if( pulses.getNumElems() >= nPulses )
                {
                    break;
                }
                    
                pulse.pulse_ID = nCurrentPulse;
                pulse.timestamp = m_TempBuffer.timestamp;
                pulse.azimuth = std::atan(point.x / point.y);
                if( (point.x <= 0) && (point.y >= 0) )
                {
                    pulse.azimuth += 180;
                }
                if( (point.x <= 0) && (point.y <= 0) )
                {
                    pulse.azimuth -= 180;
                }
                if( pulse.azimuth < 0 )
                {
                    pulse.azimuth += 360;
                }
                pulse.zenith = std::atan(std::sqrt(std::pow(point.x, 2) + 
                                std::pow(point.y, 2)) / point.z);
                if( pulse.zenith < 0 )
                {
                    pulse.zenith += 180;
                }
                    
                pulse.scanline = m_TempBuffer.row;
                pulse.scanline_Idx = m_TempBuffer.column;
                pulse.x_Idx = point.x;
                pulse.y_Idx = point.y;
                pulse.pts_start_idx = points.getNumElems();
                pulse.number_of_returns = 0; // can't trust m_TempBuffer.target_count;
                                            // increment below
                pulses.push(&pulse);
                nCurrentPulse++;
            }
            
            points.push(&point);
            
            SRieglRDBPulse *pLastPulse = pulses.getLastElement();
            if( pLastPulse != NULL )
            {
                pLastPulse->number_of_returns++;
            }
            
        }

        // build tuple
        PyArrayObject *pPulses = pulses.getNumpyArray(RieglPulseFields);
        PyArrayObject *pPoints = points.getNumpyArray(RieglPointFields);
        PyObject *pTuple = PyTuple_Pack(2, pPulses, pPoints);
        Py_DECREF(pPulses);
        Py_DECREF(pPoints);

        return pTuple;
    }

private:
    RDBContext *m_pContext;
    RDBPointcloudQuerySelect *m_pQuerySelect;
    Py_ssize_t m_nPulsesToIgnore;
    RieglRDBBuffer m_TempBuffer;
};

static void 
PyRieglRDBFile_dealloc(PyRieglRDBFile *self)
{
    free(self->pszFilename);
    if( self->pQuerySelect != NULL )
    {
        rdb_pointcloud_query_select_delete(self->pContext, &self->pQuerySelect);
    }
    if( self->pPointCloud != NULL )
    {
        rdb_pointcloud_delete(self->pContext, &self->pPointCloud);
    }
    if( self->pContext != NULL )
    {
        rdb_context_delete(&self->pContext);
    }
    
    Py_XDECREF(self->pHeader);

    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Processes psz until a '.' is found
// and returns string as an int. 
// Updates psz to point to first char after the '.'
static int GetAVersionPart(RDBString *psz)
{
RDBString pCurr = *psz;

    while( *pCurr != '.' )
    {
        pCurr++;
    }
    
    char *pTemp = strndup(*psz, pCurr - (*psz));
    int nResult = atoi(pTemp);
    free(pTemp);
    
    pCurr++; // go past dot
    *psz = pCurr; // store back to caller
    
    return nResult;
}

static int 
PyRieglRDBFile_init(PyRieglRDBFile *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname ) )
    {
        return -1;
    }
    
    // get context
    // TODO: log level and log path as an option?
    CHECKRESULT_FILE(rdb_context_new(&self->pContext, "", ""), -1)
    
    // check we are running against the same version of the library we were compiled
    // against
    RDBString pszVersionString;
    CHECKRESULT_FILE(rdb_library_version(self->pContext, &pszVersionString), -1)
    
    RDBString psz = pszVersionString;
    int nMajor = GetAVersionPart(&psz);
    int nMinor = GetAVersionPart(&psz);
    if( (nMajor != RIEGL_RDB_MAJOR) || (nMinor != RIEGL_RDB_MINOR) )
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC->error, "Mismatched libraries - RDB lib differs in version "
            "Was compiled against version %d.%d. Now running with %d.%d (version string: '%s')\n", 
            RIEGL_RDB_MAJOR, RIEGL_RDB_MINOR, nMajor, nMinor, pszVersionString);
        return -1;
    }
    
    // open file
    // need a RDBPointcloudOpenSettings TODO: set from options
    RDBPointcloudOpenSettings *pSettings;
    CHECKRESULT_FILE(rdb_pointcloud_open_settings_new(self->pContext, &pSettings), -1)
    
    CHECKRESULT_FILE(rdb_pointcloud_new(self->pContext, &self->pPointCloud), -1)
    
    CHECKRESULT_FILE(rdb_pointcloud_open(self->pContext, self->pPointCloud,
                    pszFname, pSettings), -1)
                    
    /*uint32_t count;
    RDBString list;
    CHECKRESULT_FILE(rdb_pointcloud_point_attributes_list(self->pContext, self->pPointCloud,
                        &count, &list), -1)
    for( uint32_t i = 0; i < count; i++ )
    {
        fprintf(stderr, "%d %s\n", i, list);
        list = list + strlen(list) + 1;
    }*/
    // read in header ("metadata" in rdblib speak)
    self->pHeader = PyDict_New();
    uint32_t count;
    RDBString list;
    CHECKRESULT_FILE(rdb_pointcloud_meta_data_list(self->pContext, self->pPointCloud,
                        &count, &list), -1)
    RDBString value;
    for( uint32_t i = 0; i < count; i++ )
    {
        CHECKRESULT_FILE(rdb_pointcloud_meta_data_get(self->pContext, self->pPointCloud,
                        list, &value), -1)
        // TODO: decode json?
#if PY_MAJOR_VERSION >= 3
        PyObject *pString = PyUnicode_FromString(value);
#else
        PyObject *pString = PyString_FromString(value);
#endif
        PyDict_SetItemString(self->pHeader, list, pString);
        Py_DECREF(pString);
        /*fprintf(stderr, "%d %s %s\n", i, list, value);*/
        list = list + strlen(list) + 1;
    }

    // take a copy of the filename so we can re
    // create the file pointer.
    self->pszFilename = strdup(pszFname);
    
    self->pQuerySelect = NULL;
    self->nCurrentPulse = 0;
    self->bFinishedReading = false;

    return 0;
}

static PyObject *PyRieglRDBFile_readData(PyRieglRDBFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;
    
    Py_ssize_t nPulsesToIgnore = 0;
    if( self->pQuerySelect == NULL )
    {
        fprintf(stderr, "New start\n");
        // have no current selection - create one
        CHECKRESULT_FILE(rdb_pointcloud_query_select_new(self->pContext, 
                    self->pPointCloud,
                    0, // all nodes apparently according to querySelect.cpp
                    "",
                    &self->pQuerySelect), NULL)
        
        nPulsesToIgnore = nPulseStart;
    }
    else
    {
        // already have one
        if( nPulseStart > self->nCurrentPulse )
        {
            fprintf(stderr, "fast forward\n" );
            // past where we are up to
            nPulsesToIgnore = nPulseStart - self->nCurrentPulse;
        }
        else if( nPulseStart < self->nCurrentPulse )
        {
            fprintf(stderr, "rewind\n");
            // go back to beginning. Delete query select and start again
            CHECKRESULT_FILE(rdb_pointcloud_query_select_delete(self->pContext, &self->pQuerySelect), NULL)

            CHECKRESULT_FILE(rdb_pointcloud_query_select_new(self->pContext, 
                    self->pPointCloud,
                    0, // all nodes apparently according to querySelect.cpp
                    "",
                    &self->pQuerySelect), NULL)
            nPulsesToIgnore = nPulseStart;
        }
    }
    
    RiegRDBReader reader(self->pContext, self->pQuerySelect);
    reader.setPulsesToIgnore(nPulsesToIgnore);
    
    // will be NULL on error
    PyObject *pResult = reader.readData(nPulses, 
                self->nCurrentPulse, self->bFinishedReading);
    
    return pResult;
}

static PyObject *PyRieglRDBFile_readWaveforms(PyRieglRDBFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd;
    if( !PyArg_ParseTuple(args, "nn:readWaveforms", &nPulseStart, &nPulseEnd ) )
        return NULL;

    Py_RETURN_NONE;
}

// Returns false and sets Python exception if error code set
static bool CheckRDBResult(RDBResult code, RDBContext *pContext)
{
    if( code != RDB_SUCCESS )
    {
        int32_t   errorCode(0);
        RDBString text("Unable to create context");
        RDBString details("");
    
        if( pContext != NULL )
        {
            rdb_context_get_last_error(pContext, &errorCode, &text, &details);
        }
        
        PyErr_Format(GETSTATE_FC->error, "Error from RDBLib: %s. Details: %s\n",
               text, details); 
        return false;
    }
    
    return true;
}

/* Table of methods */
static PyMethodDef PyRieglRDBFile_methods[] = {
    {"readData", (PyCFunction)PyRieglRDBFile_readData, METH_VARARGS, NULL},
    {"readWaveforms", (PyCFunction)PyRieglRDBFile_readWaveforms, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyObject *PyRieglRDBFile_getFinished(PyRieglRDBFile *self, void *closure)
{
    if( self->bFinishedReading )
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject *PyRieglRDBFile_getPulsesRead(PyRieglRDBFile *self, void *closure)
{
    return PyLong_FromSsize_t(self->nCurrentPulse);
}

static PyObject *PyRieglRDBFile_getHeader(PyRieglRDBFile *self, void *closure)
{
    Py_INCREF(self->pHeader);
    return self->pHeader;
}

/* get/set */
static PyGetSetDef PyRieglRDBFile_getseters[] = {
    {(char*)"finished", (getter)PyRieglRDBFile_getFinished, NULL, (char*)"Get Finished reading state", NULL}, 
    {(char*)"pulsesRead", (getter)PyRieglRDBFile_getPulsesRead, NULL, (char*)"Get number of pulses read", NULL},
    {(char*)"header", (getter)PyRieglRDBFile_getHeader, NULL, (char*)"Get header as a dictionary", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyRieglRDBFileType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_rieglrdb.RDBFile",         /*tp_name*/
    sizeof(PyRieglRDBFile),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyRieglRDBFile_dealloc, /*tp_dealloc*/
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
    "Riegl RDB File object",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyRieglRDBFile_methods,             /* tp_methods */
    0,             /* tp_members */
    PyRieglRDBFile_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyRieglRDBFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};
#if PY_MAJOR_VERSION >= 3

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__rieglrdb(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_rieglrdb(void)
#endif
{
    PyObject *pModule;
    struct RieglRDBState *state;

    /* initialize the numpy stuff */
    import_array();
    /* same for pylidar functions */
    pylidar_init();

#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("_rieglrdb", module_methods);
#endif
    if( pModule == NULL )
        INITERROR;

    state = GETSTATE(pModule);

    /* Create and add our exception type */
    state->error = PyErr_NewException("_rieglrdb.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    PyModule_AddObject(pModule, "error", state->error);

    /* Scan file type */
    PyRieglRDBFileType.tp_new = PyType_GenericNew;
    if( PyType_Ready(&PyRieglRDBFileType) < 0 )
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PyRieglRDBFileType);
    PyModule_AddObject(pModule, "RDBFile", (PyObject *)&PyRieglRDBFileType);

#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
