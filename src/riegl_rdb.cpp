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
#include <memory>
#include <unordered_map>
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

#ifdef _MSC_VER
// not available with MSVC
// see https://bugs.libssh.org/T112
char *strndup(const char *s, size_t n)
{
    char *x = NULL;

    if (n + 1 < n) {
        return NULL;
    }

    x = (char*)malloc(n + 1);
    if (x == NULL) {
        return NULL;
    }

    memcpy(x, s, n);
    x[n] = '\0';

    return x;
}

#endif

/* Structure for pulses */
typedef struct {
    npy_uint64 pulse_ID;
    npy_uint64 timestamp;
    float azimuth;
    float zenith;
    npy_uint32 scanline; // abs() of riegl.row 
    npy_uint16 scanline_Idx; // abs() of riegl.column
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
    double timestamp; // riegl.timestamp (convert s->ns and to uint64 later)
    npy_int32 deviation; // riegl.deviation
    npy_uint16 classification; // riegl.class
    double reflectance; // riegl.reflectance
    double amplitude; // riegl.amplitude
    double xyz[3]; // riegl.xyz
    
    // pulses
    npy_int32 row;  // riegl.row - can be negative
    npy_int32 column; // riegl.column ditto
    
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

// class that tracks an incomplete pulse.
// The records in a .rbdx file are in a 'random' order
// they have been optimised for display in some sort of order that
// has a low res visualisation first, then as you read through 
// the file you get more density.

// This is a bit of a pain for us since you get the individual records
// for a pulse all over the place. We track everything by the riegl.id 
// of the first record for a pulse (riedl.id - reigl.target_index - 1) for
// a given record. This class allows you to add more records for a pulse
// and work out when you have them all and get an array of these records.

// PyRieglRDBDataBuffer uses an unordered_map of these based on the riegl.id
// of the first record of a pulse.
class PyRieglRDBPulseTracker
{
public:
    PyRieglRDBPulseTracker(npy_uint8 nrecords)
    {
        m_nRecords = nrecords;
        m_pRecords = new RieglRDBBuffer[nrecords];
        m_pSet = new bool[nrecords];
        for( npy_uint8 i = 0; i < nrecords; i++ )
        {
            m_pSet[i] = false;
        }
    }
    ~PyRieglRDBPulseTracker()
    {
        delete[] m_pRecords;
        delete[] m_pSet;
    }
    
    void setRecord(npy_uint8 idx, RieglRDBBuffer *pRecord)
    {
        if( idx < m_nRecords )
        {
            memcpy(&m_pRecords[idx], pRecord, sizeof(RieglRDBBuffer));
            m_pSet[idx] = true;
        }
    }
    
    // returns m_pRecords if we have them all, otherwise NULL
    RieglRDBBuffer* allSet()
    {
        for( npy_uint8 i = 0; i < m_nRecords; i++ )
        {
            if( !m_pSet[i] )
            {
                return NULL;
            }
        }
        //fprintf(stderr, "Got complete records %d\n", (int)m_nRecords);
        return m_pRecords;
    }
    
    npy_uint8 getNRecords()
    {
        return m_nRecords;
    }
    
    npy_uint8 getNFound()
    {
        npy_uint8 found = 0;
        for( npy_uint8 i = 0; i < m_nRecords; i++ )
        {
            if( m_pSet[i] )
            {
                found++;
            }
        }
        return found;
    }

    // if there are incomplete pulses, shuffle everything down 
    // and update m_nRecords
    // returns pointer to buffer
    RieglRDBBuffer* compress()
    {
        while( allSet() == NULL )
        {
            for( npy_uint8 i = 0; i < m_nRecords; i++ )
            {
                if( !m_pSet[i] )
                {
                    shuffleDownOne(i);
                }
            }
        }
        return m_pRecords;
    }
    
    void dump()
    {
        for( npy_uint8 i = 0; i < m_nRecords; i++ )
        {
            fprintf(stderr, "Dump %d %ld %d\n", (int)m_pSet[i], m_pRecords[i].id, (int)m_pRecords[i].target_count);
        }
    }

private:
    void shuffleDownOne(npy_uint8 clobberIdx)
    {
        for( npy_uint8 i = clobberIdx; i < (m_nRecords-1); i++ )
        {
            memcpy(&m_pRecords[i], &m_pRecords[i+1], sizeof(RieglRDBBuffer));
            m_pSet[i] = m_pSet[i+1];
        }
        m_nRecords--;
    }

    RieglRDBBuffer *m_pRecords;
    npy_uint8 m_nRecords;
    bool *m_pSet;
};

// class that wraps an array of RieglRDBBuffer s
// so we can read in a whole lot at once. 
class PyRieglRDBDataBuffer
{
public:
    PyRieglRDBDataBuffer(RDBContext *pContext, RDBPointcloudQuerySelect *pQuerySelect)
    {
        m_pContext = pContext;
        m_pQuerySelect = pQuerySelect;
        reset();
    }
    
    void reset()
    {
        m_nElementsInBuffer = 0;
        m_nCurrentIdx = 0;
        m_bEOF = false;
    }
    
    // so we can update it if we have to rewind etc
    void setQuerySelect(RDBPointcloudQuerySelect *pQuerySelect)
    {
        m_pQuerySelect = pQuerySelect;
        m_pulseTracker.clear(); // clear the tracker as nothing will make sense now
        reset();
    }

    // reset librdb to read into the start of our m_TempBuffer
    bool resetBuffers()
    {
        // NOTE: this must match definition of RieglRDBBuffer
        // I'm using the info from attribues.h rather than strings
        //  - not sure what is best practice
        // point
        CHECKBIND_READER(RDB_RIEGL_ID.name, RDBDataTypeUINT64, &m_buffer[0].id)
        CHECKBIND_READER(RDB_RIEGL_TIMESTAMP.name, RDBDataTypeDOUBLE, &m_buffer[0].timestamp)
        CHECKBIND_READER(RDB_RIEGL_DEVIATION.name, RDBDataTypeINT32, &m_buffer[0].deviation)
        CHECKBIND_READER(RDB_RIEGL_CLASS.name, RDBDataTypeUINT16, &m_buffer[0].classification)
        CHECKBIND_READER(RDB_RIEGL_REFLECTANCE.name, RDBDataTypeDOUBLE, &m_buffer[0].reflectance)
        CHECKBIND_READER(RDB_RIEGL_AMPLITUDE.name, RDBDataTypeDOUBLE, &m_buffer[0].amplitude)
        CHECKBIND_READER(RDB_RIEGL_XYZ.name, RDBDataTypeDOUBLE, &m_buffer[0].xyz)
        
        // these 2 don't appear to be documented, but are in there
        CHECKBIND_READER("riegl.row", RDBDataTypeINT32, &m_buffer[0].row);
        CHECKBIND_READER("riegl.column", RDBDataTypeINT32, &m_buffer[0].column);
        
        CHECKBIND_READER(RDB_RIEGL_TARGET_INDEX.name, RDBDataTypeUINT8, &m_buffer[0].target_index)
        CHECKBIND_READER(RDB_RIEGL_TARGET_COUNT.name, RDBDataTypeUINT8, &m_buffer[0].target_count)
        
        return true;
    }
    
    bool read(uint32_t nElements=nInitSize)
    {
        reset();
        if( !resetBuffers() )
        {
            return false;
        }
        uint32_t processed = 0;
        if( rdb_pointcloud_query_select_next(
                m_pContext, m_pQuerySelect, nElements,
                &processed) == 0 )
        {
            // 0 means 'end of file' apparently
            m_bEOF = true;
        }
        if( processed == 0 )
        {
            m_bEOF = true;
        }
        
        m_nElementsInBuffer = processed;
        
        return true;
    }
    
    
    bool move()
    {
        m_nCurrentIdx++;
        if( m_nCurrentIdx >= m_nElementsInBuffer )
        {
            bool bFirstRead = (m_nElementsInBuffer == 0);
            //fprintf(stderr, "refilling 1 %d %d\n", m_nCurrentIdx, m_nElementsInBuffer);
            if( !read() )
            {
                return false;
            }
            if( bFirstRead )
            {
                // we haven't done a read() before now, go to element 1
                m_nCurrentIdx = 1;
            }
            else
            {
                // have done a read() before, go back to start
                m_nCurrentIdx = 0;
            }
        }
        return true;
    }
    
    RieglRDBBuffer *getCurrent()
    {
        if( m_nElementsInBuffer == 0 )
        {
            if( !read() )
            {
                return NULL;
            }
        }
        return &m_buffer[m_nCurrentIdx];
    }

    bool eof()
    {
        return m_bEOF;
    }
    
    // support for tracking incomplete pulses (see PyRieglRDBPulseTracker, above)
    
    // return pointer to buffer if have all the records for a pulse
    RieglRDBBuffer* addCurrentToTracker()
    {
        RieglRDBBuffer *pCurrEl = getCurrent();
        if( pCurrEl == NULL )
        {
            return NULL;
        }
        // target_index is 1-based.
        // find the id of the first record of this pulse
        // (should be in order)
        npy_uint64 startIdx = pCurrEl->id - (pCurrEl->target_index - 1);
        auto got = m_pulseTracker.find(startIdx);
        
        if( got == m_pulseTracker.end() )
        {
            if( pCurrEl->target_count == 1 )
            {
                // can short circuit the rest and return pCurrEl
                //fprintf(stderr, "Just one el %ld\n", startIdx);
                return pCurrEl;
            }
            else
            {
                //fprintf(stderr, "Starting new one for %ld %d %ld %d\n", startIdx, (int)pCurrEl->target_count, pCurrEl->id, (int)pCurrEl->target_index);
                PyRieglRDBPulseTracker *pTracker = new PyRieglRDBPulseTracker(pCurrEl->target_count);
                pTracker->setRecord(pCurrEl->target_index - 1, pCurrEl);
                std::unique_ptr<PyRieglRDBPulseTracker> ap(pTracker);
                
                m_pulseTracker.insert(std::pair<npy_uint64, std::unique_ptr<PyRieglRDBPulseTracker> >(startIdx, std::move(ap)));
                return NULL;
            }
        }
        else
        {
            //fprintf(stderr, "Adding record for %ld\n", startIdx);
            got->second.get()->setRecord(pCurrEl->target_index - 1, pCurrEl);
            return got->second.get()->allSet();
        }
    }
    
    // when finished processing a complete pulse, 
    // call to release memory
    void removeCurrentFromTracker()
    {
        RieglRDBBuffer *pCurrEl = getCurrent();
        npy_uint64 startIdx = pCurrEl->id - (pCurrEl->target_index - 1);
        // TODO: exception raised???
        m_pulseTracker.erase(startIdx);
        //fprintf(stderr, "removing %ld\n", startIdx);
    }
    
    RieglRDBBuffer* getNextRemainderInTracker(npy_uint8 &nRecords)
    {
        auto itr = m_pulseTracker.begin();
        if( itr == m_pulseTracker.end() )
        {
            return NULL;
        }
        
        RieglRDBBuffer *pBuffer = itr->second.get()->compress();
        nRecords = itr->second.get()->getNRecords();
        return pBuffer;
    }
    
    bool removeRemainderFromTracker(npy_uint64 startIdx)
    {
        if( m_pulseTracker.erase(startIdx) == 0 )
        {
            return false;
        }
        return true;
    }
    
    size_t getCountOfRemaindersInTracker()
    {
        return m_pulseTracker.size();
    }
    
    // for debugging
    void dumpRemainderInTracker()
    {
        npy_uint64 tot = 0;
        for(auto itr = m_pulseTracker.begin(); itr != m_pulseTracker.end(); itr++ )
        {
            npy_uint8 missing = itr->second.get()->getNRecords() - itr->second.get()->getNFound();
            tot += missing;
            fprintf(stderr, "Remainder id = %ld count = %d\n", itr->first, missing );
        }
        fprintf(stderr, "Total %ld\n", tot);
    }
    
    size_t count(npy_uint64 startIdx)
    {
        return m_pulseTracker.count(startIdx);
    }
    
private:
    RDBContext *m_pContext;
    RDBPointcloudQuerySelect *m_pQuerySelect;
    RieglRDBBuffer m_buffer[nInitSize];
    uint32_t m_nElementsInBuffer;
    uint32_t m_nCurrentIdx;
    bool m_bEOF;
    
    // link the start riegl.id for a pulse with the points for that pulse
    std::unordered_map<npy_uint64, std::unique_ptr<PyRieglRDBPulseTracker> > m_pulseTracker;
};

typedef struct
{
    PyObject_HEAD
    char *pszFilename; // so we can re-create the file obj if needed
    RDBContext *pContext;
    RDBPointcloud *pPointCloud;
    RDBPointcloudQuerySelect *pQuerySelect; // != NULL if we currently have one going
    PyRieglRDBDataBuffer *pBuffer; // object for buffering reads on pQuerySelect
    Py_ssize_t nCurrentPulse;  // if pQuerySelect != NULL this is the pulse number we are up to
    bool bFinishedReading;
    PyObject *pHeader;  // dictionary with header info

} PyRieglRDBFile;

static const char *SupportedDriverOptions[] = {"DUMP_FIELDS_ON_OPEN", NULL};
static PyObject *rieglrdb_getSupportedOptions(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(SupportedDriverOptions);
}

// module methods
static PyMethodDef module_methods[] = {
    {"getSupportedOptions", (PyCFunction)rieglrdb_getSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options"},
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

class RieglRDBReader
{
public:
    RieglRDBReader(PyRieglRDBDataBuffer *pBuffer)
    {
        m_pBuffer = pBuffer;
        m_nPulsesToIgnore = 0;
    }
    
    void setPulsesToIgnore(Py_ssize_t nPulsesToIgnore)
    {
        m_nPulsesToIgnore = nPulsesToIgnore;
    }
    
    void ConvertRecordsToPointsAndPulses(RieglRDBBuffer *pCurrEl, npy_uint8 target_count,
                                            pylidar::CVector<SRieglRDBPulse> &pulses,
                                            pylidar::CVector<SRieglRDBPoint> &points,
                                            Py_ssize_t nCurrentPulse)
    {
        SRieglRDBPulse pulse;
        SRieglRDBPoint point;
    
        // create pulse
        pulse.pulse_ID = nCurrentPulse;
        pulse.timestamp = (npy_uint64)round(pCurrEl->timestamp * 1e10); // time of first record in ns
        // use first record for calculating azimuth etc (should be last record?)
        double x = pCurrEl->xyz[0];
        double y = pCurrEl->xyz[1];
        double z = pCurrEl->xyz[2];
        pulse.azimuth = (float)std::atan(x / y);
        if( (x <= 0) && (y >= 0) )
        {
            pulse.azimuth += 180;
        }
        if( (x <= 0) && (y <= 0) )
        {
            pulse.azimuth -= 180;
        }
        if( pulse.azimuth < 0 )
        {
            pulse.azimuth += 360;
        }
        pulse.zenith = (float)std::atan(std::sqrt(std::pow(x, 2) + 
                        std::pow(y, 2)) / z);
        if( pulse.zenith < 0 )
        {
            pulse.zenith += 180;
        }

        // both riegl.row and riegl.column can be positive or negative
        // the SPDV4 spec has these unsigned and we get into trouble trying
        // to import as negative values are outside the range. 
        // Generally, we have found that these attributes are either 
        // positive OR negative - not both for the same files. So we
        // abs() here to keep everything happy...
        pulse.scanline = std::abs(pCurrEl->row);
        pulse.scanline_Idx = std::abs(pCurrEl->column);
        pulse.x_Idx = x;
        pulse.y_Idx = y;
        pulse.pts_start_idx = (npy_uint32)points.getNumElems();
        pulse.number_of_returns = target_count;
        pulses.push(&pulse);

        // now points
        for( npy_uint8 i = 0; i < target_count; i++ )
        {
            // convert to ns
            point.timestamp = (npy_uint64)round(pCurrEl->timestamp * 1e10);
            point.deviation_Return = (float)pCurrEl->deviation;
            point.classification = (npy_uint8)pCurrEl->classification;
            point.range = std::sqrt(std::pow(pCurrEl->xyz[0], 2) + 
                            std::pow(pCurrEl->xyz[1], 2) + 
                            std::pow(pCurrEl->xyz[2], 2));
            point.rho_app = pCurrEl->reflectance;
            point.amplitude_Return = pCurrEl->amplitude;
            point.x = pCurrEl->xyz[0];
            point.y = pCurrEl->xyz[1];
            point.z = pCurrEl->xyz[2];
            point.return_Number = i + 1;  // 1-based

            points.push(&point);
            // next record
            pCurrEl++;
        }
            
    }                                            
                                        
    
    
    PyObject* readData(Py_ssize_t nPulses, Py_ssize_t &nCurrentPulse, bool &bFinished)
    {
        pylidar::CVector<SRieglRDBPulse> pulses(nInitSize, nGrowBy);
        pylidar::CVector<SRieglRDBPoint> points(nInitSize, nGrowBy);
        
        while( m_nPulsesToIgnore > 0 )
        {
            //fprintf(stderr, "Ignoring %ld\n", m_nPulsesToIgnore);
            // move first so we don't include the pulse we are currently on
            if( !m_pBuffer->move() )
            {
                bFinished = true;
                return NULL;
            }

            if( m_pBuffer->eof() )
            {
                // skip remainder also
                if( m_pBuffer->getCountOfRemaindersInTracker() > 0 )
                {
                    npy_uint8 nrecords;
                    RieglRDBBuffer *pEl = m_pBuffer->getNextRemainderInTracker(nrecords);
                    m_pBuffer->removeRemainderFromTracker(pEl->id - (pEl->target_index - 1));
                }
                else
                {
                    PyErr_SetString(GETSTATE_FC->error, "Got to EOF while ignoring pulses");
                    return NULL;
                }
            }

            if( m_pBuffer->getCurrent() == NULL )
            {
                // error whould be set
                bFinished = true;
                return NULL;
            }
            if( m_pBuffer->addCurrentToTracker() != NULL)
            {
                m_pBuffer->removeCurrentFromTracker();
                // we have a complete pulse 
                m_nPulsesToIgnore--;
                nCurrentPulse++;
            }
        }

        // now do the actual reading
        while( !m_pBuffer->eof() && (pulses.getNumElems() < nPulses))
        {
            if( m_pBuffer->getCurrent() == NULL )
            {
                bFinished = true;
                // error happened somewhere
                return NULL;
            }

            RieglRDBBuffer *pRecordsStart = m_pBuffer->addCurrentToTracker();
            if( pRecordsStart != NULL )
            {
                // we have a complete pulse with pRecordsStart->target_count records
                ConvertRecordsToPointsAndPulses(pRecordsStart, pRecordsStart->target_count, 
                                    pulses, points, nCurrentPulse);
            
                nCurrentPulse++;
                m_pBuffer->removeCurrentFromTracker();
            }            
            
            if( !m_pBuffer->move() )
            {
                bFinished = true;
                return NULL;
            }
        }
        
        // any remainder?
        if( m_pBuffer->eof() && (pulses.getNumElems() < nPulses) && (m_pBuffer->getCountOfRemaindersInTracker() > 0))
        {
            while( pulses.getNumElems() < nPulses )
            {
                npy_uint8 target_count = 0;
                RieglRDBBuffer *pRecordsStart = m_pBuffer->getNextRemainderInTracker(target_count);
                //fprintf(stderr, "Got remainder of %d %ld\n", (int)target_count, nCurrentPulse);
                if( pRecordsStart == NULL )
                {
                    // no more remainders
                    break;
                }
                ConvertRecordsToPointsAndPulses(pRecordsStart, target_count, 
                                    pulses, points, nCurrentPulse);
                nCurrentPulse++;
                m_pBuffer->removeRemainderFromTracker(pRecordsStart->id - (pRecordsStart->target_index - 1));
            }
        }
        
        bFinished = (m_pBuffer->eof() && (m_pBuffer->getCountOfRemaindersInTracker() == 0));

        // build tuple
        PyArrayObject *pPulses = pulses.getNumpyArray(RieglPulseFields);
        PyArrayObject *pPoints = points.getNumpyArray(RieglPointFields);
        PyObject *pTuple = PyTuple_Pack(2, pPulses, pPoints);
        Py_DECREF(pPulses);
        Py_DECREF(pPoints);

        return pTuple;
    }

private:
    PyRieglRDBDataBuffer *m_pBuffer;
    Py_ssize_t m_nPulsesToIgnore;
};

static void 
PyRieglRDBFile_dealloc(PyRieglRDBFile *self)
{
    delete self->pBuffer;
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
PyObject *pOptionDict;

    if( !PyArg_ParseTuple(args, "sO", &pszFname, &pOptionDict ) )
    {
        return -1;
    }
    
    if( !PyDict_Check(pOptionDict) )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Last parameter to init function must be a dictionary");
        return -1;
    }
    
    // parse options
    bool bDumpFields = false;
    PyObject *pDumpFieldsFlag = PyDict_GetItemString(pOptionDict, "DUMP_FIELDS_ON_OPEN");
    if( pDumpFieldsFlag != NULL )
    {
        if( !PyBool_Check(pDumpFieldsFlag) )
        {
            PyErr_SetString(GETSTATE_FC->error, "DUMP_FIELDS_ON_OPEN must be True or False");
            return -1;
        }
        if( pDumpFieldsFlag == Py_True )
        {
            bDumpFields = true;
        }
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
    
    if( bDumpFields )
    {                
        uint32_t count;
        RDBString list;
        CHECKRESULT_FILE(rdb_pointcloud_point_attributes_list(self->pContext, self->pPointCloud,
                        &count, &list), -1)
        const char *pszDataTypeNames[] = {"NONE", "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
                                "UINT64", "INT64", "SINGLE", "DOUBLE"};
        RDBPointcloudPointAttribute *attr;
        CHECKRESULT_FILE(rdb_pointcloud_point_attribute_new(self->pContext, &attr), -1)
        
        for( uint32_t i = 0; i < count; i++ )
        {
            CHECKRESULT_FILE(rdb_pointcloud_point_attributes_get(self->pContext,
                        self->pPointCloud, list, attr), -1)      
        
            uint32_t type = 0;
            CHECKRESULT_FILE(rdb_pointcloud_point_attribute_data_type(self->pContext, attr, &type), -1)
        
            fprintf(stderr, "%d %s %s\n", i, list, pszDataTypeNames[type]);
            list = list + strlen(list) + 1;
        }
        CHECKRESULT_FILE(rdb_pointcloud_point_attribute_delete(self->pContext, &attr), -1)
    }
    
    // read in header ("metadata" in rdblib speak)
    self->pHeader = PyDict_New();
    uint32_t acount;
    RDBString alist;
    CHECKRESULT_FILE(rdb_pointcloud_meta_data_list(self->pContext, self->pPointCloud,
                        &acount, &alist), -1)
    RDBString value;
    for( uint32_t i = 0; i < acount; i++ )
    {
        CHECKRESULT_FILE(rdb_pointcloud_meta_data_get(self->pContext, self->pPointCloud,
                        alist, &value), -1)
        // JSON decoding happens in Python
#if PY_MAJOR_VERSION >= 3
        PyObject *pString = PyUnicode_FromString(value);
#else
        PyObject *pString = PyString_FromString(value);
#endif
        PyDict_SetItemString(self->pHeader, alist, pString);
        Py_DECREF(pString);
        /*fprintf(stderr, "%d %s %s\n", i, list, value);*/
        alist = alist + strlen(alist) + 1;
    }

    // take a copy of the filename so we can re
    // create the file pointer.
    self->pszFilename = strdup(pszFname);
    
    self->pQuerySelect = NULL;
    self->pBuffer = NULL;
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
    //fprintf( stderr, "Pulse start %ld\n", nPulseStart);
    
    Py_ssize_t nPulsesToIgnore = 0;
    if( self->pQuerySelect == NULL )
    {
        //fprintf(stderr, "New start\n");
        // have no current selection - create one
        CHECKRESULT_FILE(rdb_pointcloud_query_select_new(self->pContext, 
                    self->pPointCloud,
                    0, // all nodes apparently according to querySelect.cpp
                    "",
                    &self->pQuerySelect), NULL)
                    
        self->pBuffer = new PyRieglRDBDataBuffer(self->pContext,
                                        self->pQuerySelect);
        
        nPulsesToIgnore = nPulseStart;
        self->nCurrentPulse = 0;
    }
    else
    {
        // already have one
        if( nPulseStart > self->nCurrentPulse )
        {
            //fprintf(stderr, "fast forward\n" );
            // past where we are up to
            nPulsesToIgnore = nPulseStart - self->nCurrentPulse;
        }
        else if( nPulseStart < self->nCurrentPulse )
        {
            //fprintf(stderr, "rewind\n");
            // go back to beginning. Delete query select and start again
            CHECKRESULT_FILE(rdb_pointcloud_query_select_delete(self->pContext, &self->pQuerySelect), NULL)

            CHECKRESULT_FILE(rdb_pointcloud_query_select_new(self->pContext, 
                    self->pPointCloud,
                    0, // all nodes apparently according to querySelect.cpp
                    "",
                    &self->pQuerySelect), NULL)

            self->pBuffer->setQuerySelect(self->pQuerySelect);
            self->nCurrentPulse = 0;
                                        
            nPulsesToIgnore = nPulseStart;
        }
    }
    
    RieglRDBReader reader(self->pBuffer);
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
