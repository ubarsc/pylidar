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

// buffer that we read the data in before copying to point/pulse structures
// NOTE: any changes must be reflected in RiegRDBReader::resetBuffers()
typedef struct
{
   // pulse
    npy_uint64 shot_ID; // rieg.shot_id
    double shot_origin[3]; // riegl.shot_origin
    double shot_direction[3]; // riegl.shot_direction
    npy_uint64 timestamp; // riegl.timestamp
   
    // points
    npy_uint64 id; // riegl.id
    npy_uint16 classification; // riegl.class
    double range; // riegl.range
    double reflectance; // riegl.reflectance
    double amplitude; // riegl.amplitude
    double xyz[3]; // riegl.xyz_socs

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
        // pulse
        //CHECKBIND_READER("riegl.shot_id", RDBDataTypeUINT64, &m_TempBuffer[0].shot_ID)
        //CHECKBIND_READER("riegl.shot_origin", RDBDataTypeDOUBLE, &m_TempBuffer[0].shot_origin)
        //CHECKBIND_READER("riegl.shot_direction", RDBDataTypeDOUBLE, &m_TempBuffer[0].shot_direction)
        CHECKBIND_READER("riegl.timestamp", RDBDataTypeUINT64, &m_TempBuffer[0].timestamp)

        // point
        CHECKBIND_READER("riegl.id", RDBDataTypeUINT64, &m_TempBuffer[0].id)
        CHECKBIND_READER("riegl.class", RDBDataTypeUINT16, &m_TempBuffer[0].classification)
        //CHECKBIND_READER(RDB_RIEGL_RANGE.name, RDBDataTypeDOUBLE, &m_TempBuffer[0].range)
        CHECKBIND_READER("riegl.reflectance", RDBDataTypeDOUBLE, &m_TempBuffer[0].reflectance)
        CHECKBIND_READER("riegl.amplitude", RDBDataTypeDOUBLE, &m_TempBuffer[0].amplitude)
        CHECKBIND_READER("riegl.xyz", RDBDataTypeDOUBLE, &m_TempBuffer[0].xyz)
        
        return true;
    }
    
    bool readData(Py_ssize_t nPulses)
    {
        // reset to beginning of our temp space
        if( !resetBuffers() )
        {
            return false;
        }
        
        uint32_t processed = 0;
        CHECKRESULT_READER(rdb_pointcloud_query_select_next(
                    m_pContext, m_pQuerySelect, nTempSize, &processed), false);
                    
        for( uint32_t i = 0; i < processed; i++ )
        {
            fprintf(stderr, "id %" PRIu64 " %f\n", m_TempBuffer[i].id, m_TempBuffer[i].reflectance);
        }
                    
        return true;
    }

private:
    RDBContext *m_pContext;
    RDBPointcloudQuerySelect *m_pQuerySelect;
    Py_ssize_t m_nPulsesToIgnore;
    RieglRDBBuffer m_TempBuffer[nTempSize];
};

typedef struct
{
    PyObject_HEAD
    char *pszFilename; // so we can re-create the file obj if needed
    RDBContext *pContext;
    RDBPointcloud *pPointCloud;
    RDBPointcloudQuerySelect *pQuerySelect; // != NULL if we currently have one going
    Py_ssize_t nCurrentPulse;  // if pQuerySelect != NULL this is the pulse number we are up to

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
                    
    uint32_t count;
    RDBString list;
    CHECKRESULT_FILE(rdb_pointcloud_point_attributes_list(self->pContext, self->pPointCloud,
                        &count, &list), -1)
    for( uint32_t i = 0; i < count; i++ )
    {
        fprintf(stderr, "%d %s\n", i, list);
        list = list + strlen(list) + 1;
    }
    

    // take a copy of the filename so we can re
    // create the file pointer.
    self->pszFilename = strdup(pszFname);
    
    self->pQuerySelect = NULL;
    self->nCurrentPulse = 0;

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
    if( !reader.readData(nPulses) )
    {
        return NULL;
    }

    Py_RETURN_NONE;
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

/* get/set */
static PyGetSetDef PyRieglRDBFile_getseters[] = {
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
