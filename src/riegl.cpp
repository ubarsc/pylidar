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
#include "numpy/npy_math.h"
#include "pylvector.h"
#include "pylmatrix.h"

#include <riegl/scanlib.hpp>
#include <cmath>
#include <limits>
#include "fwifc.h"

static const int nGrowBy = 10000;
static const int nInitSize = 256*256;

/* An exception object for this module */
/* created in the init function */
struct RieglState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct RieglState*)PyModule_GetState(m))
#define GETSTATE_FC GETSTATE(PyState_FindModule(&moduledef))
#else
#define GETSTATE(m) (&_state)
#define GETSTATE_FC (&_state)
static struct RieglState _state;
#endif

/* Structure for pulses */
typedef struct {
    npy_uint64 pulse_ID;
    npy_uint64 timestamp;
    npy_uint8 prism_facet;
    float azimuth;
    float zenith;
    npy_uint32 scanline;
    npy_uint16 scanline_Idx;
    double x_Idx;
    double y_Idx;
    double x_Origin;
    double y_Origin;
    double z_Origin;
    npy_uint32 pts_start_idx;
    npy_uint8 number_of_returns;
    npy_uint32 wfm_start_idx;
    npy_uint8 number_of_waveform_samples;
} SRieglPulse;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPulseFields[] = {
    CREATE_FIELD_DEFN(SRieglPulse, pulse_ID, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, timestamp, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, prism_facet, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, azimuth, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, zenith, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, scanline, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, scanline_Idx, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, y_Idx, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, x_Idx, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, x_Origin, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, y_Origin, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, z_Origin, 'f'),
    CREATE_FIELD_DEFN(SRieglPulse, pts_start_idx, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, number_of_returns, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, wfm_start_idx, 'u'),
    CREATE_FIELD_DEFN(SRieglPulse, number_of_waveform_samples, 'u'),
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
} SRieglPoint;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglPointFields[] = {
    CREATE_FIELD_DEFN(SRieglPoint, return_Number, 'u'),
    CREATE_FIELD_DEFN(SRieglPoint, timestamp, 'u'),
    CREATE_FIELD_DEFN(SRieglPoint, deviation_Return, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, classification, 'u'),
    CREATE_FIELD_DEFN(SRieglPoint, range, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, rho_app, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, amplitude_Return, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, x, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, y, 'f'),
    CREATE_FIELD_DEFN(SRieglPoint, z, 'f'),
    {NULL} // Sentinel
};

/* Structure for waveform Info */
typedef struct {
    npy_uint16 number_of_waveform_received_bins;
    npy_uint16 range_to_waveform_start;
    npy_uint64 received_start_idx;
    npy_uint8  channel;
    float      receive_wave_gain;
    float      receive_wave_offset;
} SRieglWaveformInfo;

/* field info for CVector::getNumpyArray */
static SpylidarFieldDefn RieglWaveformInfoFields[] = {
    CREATE_FIELD_DEFN(SRieglWaveformInfo, number_of_waveform_received_bins, 'u'),
    CREATE_FIELD_DEFN(SRieglWaveformInfo, range_to_waveform_start, 'u'),
    CREATE_FIELD_DEFN(SRieglWaveformInfo, received_start_idx, 'u'),
    CREATE_FIELD_DEFN(SRieglWaveformInfo, channel, 'u'),
    CREATE_FIELD_DEFN(SRieglWaveformInfo, receive_wave_gain, 'f'),
    CREATE_FIELD_DEFN(SRieglWaveformInfo, receive_wave_offset, 'f'),
    {NULL} // Sentinel
};

// This class is the main reader. It reads the points
// and pulses in chunks from the datastream.
class RieglReader : public scanlib::pointcloud
{
public:
    RieglReader(pylidar::CMatrix<double> *pRotationMatrix, pylidar::CMatrix<double> *pMagneticMatrix) : 
        scanlib::pointcloud(false), 
        m_nTotalPulsesReadFile(0),
        m_nPulsesToIgnore(0),
        m_Pulses(nInitSize, nGrowBy),
        m_Points(nInitSize, nGrowBy),
        m_scanline(0),
        m_scanlineIdx(0),
        m_pRotationMatrix(pRotationMatrix),
        m_pMagneticMatrix(pMagneticMatrix)
    {
    }

    void setPulsesToIgnore(Py_ssize_t nPulsesToIgnore)
    {
        m_nPulsesToIgnore = nPulsesToIgnore;
    }

    Py_ssize_t getNumPulsesReadFile()
    {
        return m_nTotalPulsesReadFile;
    }

    Py_ssize_t getNumPulsesRead()
    {
        return m_Pulses.getNumElems();
    }

    Py_ssize_t getNumPointsRead()
    {
        return m_Points.getNumElems();
    }

    void removeLowerPulses(Py_ssize_t n)
    {
        if( getNumPulsesRead() > 0 )
        {
            Py_ssize_t nPoints = 0;
            while(n > 0)
            {
                SRieglPulse *pPulse = m_Pulses.getElem(n - 1);
                if( (pPulse != NULL ) && ( pPulse->pts_start_idx > 0 ) )
                {
                    nPoints = pPulse->pts_start_idx;
                    break;
                }
                n--;
            }
            if( nPoints > 0 )
            {
                m_Points.removeFront(nPoints);
            }
            m_Pulses.removeFront(n);
            renumberPointIdxs();
        }
    }

    npy_uint64 getFirstPointIdx()
    {
        npy_uint32 idx = 0;
        npy_intp n = 0;
        while( n < m_Pulses.getNumElems() )
        {
            SRieglPulse *p = m_Pulses.getElem(n);
            if( p->pts_start_idx > 0 )
            {
                idx = p->pts_start_idx;
                break;
            }
            n++;
        }
        return idx;
    }

    void renumberPointIdxs()
    {
        npy_uint64 nPointIdx = getFirstPointIdx();
        if( nPointIdx == 0 )
            return;
        // reset all the pointStartIdx fields in the pulses to match
        // the array of points
        for( npy_intp n = 0; n < m_Pulses.getNumElems(); n++ )
        {
            SRieglPulse *pPulse = m_Pulses.getElem(n);
            if( pPulse->number_of_returns > 0 )
                pPulse->pts_start_idx -= nPointIdx;
        }
    }

    PyArrayObject *getPulses(Py_ssize_t n, Py_ssize_t *pPointIdx)
    {
        pylidar::CVector<SRieglPulse> *lower = m_Pulses.splitLower(n);
        // record the index of the next point
        *pPointIdx = 0;
        while( n > 0 )
        {
            if( n <= lower->getNumElems() )
            {
                SRieglPulse *pPulse = lower->getElem(n - 1);
                if( pPulse->pts_start_idx > 0 )
                {
                    *pPointIdx = (pPulse->pts_start_idx + pPulse->number_of_returns);
                    break;
                }
            }
            n--;
        }

        PyArrayObject *p = lower->getNumpyArray(RieglPulseFields);
        delete lower; // linked mem now owned by numpy
        renumberPointIdxs();
        return p;
    }

    PyArrayObject *getPoints(Py_ssize_t n)
    {
        pylidar::CVector<SRieglPoint> *lower = m_Points.splitLower(n);
        //fprintf(stderr, "points %ld %ld %ld\n", m_Points.getNumElems(), lower->getNumElems(), n);
        PyArrayObject *p = lower->getNumpyArray(RieglPointFields);
        delete lower; // linked mem now owned by numpy
        return p;
    }

    // updates the waveform info for a given pulse
    // called from readWaveforms() when called in turn from riegl_readData
    void setWaveformInfo(Py_ssize_t n, npy_uint32 wfm_start_idx, npy_uint8 number_of_waveform_samples)
    {
        SRieglPulse *pPulse = m_Pulses.getElem(n);
        pPulse->wfm_start_idx = wfm_start_idx;
        pPulse->number_of_waveform_samples = number_of_waveform_samples;
    }

protected:
    // This call is invoked for every pulse, even if there is no return
    void on_shot()
    {
        m_scanlineIdx++;
        m_nTotalPulsesReadFile++;

        if( m_nPulsesToIgnore > 0 )
        {
            m_nPulsesToIgnore--;
            return;
        }

        SRieglPulse pulse;
        pulse.pulse_ID = m_nTotalPulsesReadFile;
        // convert from seconds to ns
        pulse.timestamp = time_sorg * 1e9 + 0.5;
        pulse.prism_facet = facet;
        
        // do matrix transform and store rotated direction vector
        double beam_direction_t[3];
        applyTransformation(beam_direction[0], beam_direction[1], beam_direction[2], 0.0,
                            &beam_direction_t[0], &beam_direction_t[1], &beam_direction_t[2]);        
        
        // Get spherical coordinates
        double magnitude = std::sqrt(beam_direction_t[0] * beam_direction_t[0] + \
                           beam_direction_t[1] * beam_direction_t[1] + \
                           beam_direction_t[2] * beam_direction_t[2]);
        double shot_zenith = std::acos(beam_direction_t[2]/magnitude) * 180.0 / pi;
        double shot_azimuth = std::atan2(beam_direction_t[0],beam_direction_t[1]) * 180.0 / pi;      
        if( beam_direction_t[0] < 0 )
        {
            shot_azimuth += 360.0;            
        }

        pulse.azimuth = shot_azimuth;
        pulse.zenith = shot_zenith;
        pulse.scanline = m_scanline;
        pulse.scanline_Idx = m_scanlineIdx;
        
        // do matrix transform and store result
        applyTransformation(beam_origin[0], beam_origin[1], beam_origin[2], 1.0,
                            &pulse.x_Origin, &pulse.y_Origin, &pulse.z_Origin);

        // point idx - start with 0
        // updated when we get a point below.
        pulse.pts_start_idx = 0;
        pulse.number_of_returns = 0;

        // updated by setWaveformInfo()
        pulse.wfm_start_idx = 0;
        pulse.number_of_waveform_samples = 0;

        // set these to zero for now. 
        // not sure if we should even have them?
        pulse.x_Idx = 0;
        pulse.y_Idx = 0;

        m_Pulses.push(&pulse);
    }

    // overridden from pointcloud class
    void on_echo_transformed(echo_type echo)
    {
        if( m_nPulsesToIgnore > 0 )
        {
            // if we aren't reading pulses, we should ignore
            // points also
            return;
        }
        // we assume that this point will be
        // connected to the last pulse...
        SRieglPulse *pPulse = m_Pulses.getLastElement();
        if(pPulse == NULL)
        {
            // removed throw since this seems to happen when
            // the file pointer is reset. Ignore for now.
            //throw scanlib::scanlib_exception("Point before Pulse.");
            return;
        }
        if(pPulse->number_of_returns == 0)
        {
            // note: haven't pushed point yet
            pPulse->pts_start_idx = m_Points.getNumElems();
        }
        pPulse->number_of_returns++;

        SRieglPoint point;

        // the current echo is always indexed by target_count-1.
        scanlib::target& current_target(targets[target_count-1]);

        point.return_Number = target_count;
        point.timestamp = current_target.time * 1e9 + 0.5;
        point.deviation_Return = current_target.deviation;
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

        // Rescale reflectance and amplitude from dB to papp
        point.rho_app = std::pow(10.0, current_target.reflectance / 10.0);
        point.amplitude_Return = current_target.amplitude;

        // apply transform and store result
        applyTransformation(current_target.vertex[0], current_target.vertex[1], current_target.vertex[2], 1.0,
                            &point.x, &point.y, &point.z);

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

    void applyTransformation(double a, double b, double c, double d, double *pX, double *pY, double *pZ)
    {
        // test
        if( m_pRotationMatrix != NULL )
        {
            
             pylidar::CMatrix<double> input(4, 1);
             input.set(0, 0, a);
             input.set(1, 0, b);
             input.set(2, 0, c);
             input.set(3, 0, d); // apply transformation (1) or rotation only (0)
             pylidar::CMatrix<double> transOut = m_pRotationMatrix->multiply(input);
             a = transOut.get(0, 0);
             b = transOut.get(1, 0);
             c = transOut.get(2, 0);
             d = transOut.get(3, 0);
        }       
        if( m_pMagneticMatrix != NULL )
        {
            pylidar::CMatrix<double> input(4, 1);
            input.set(0, 0, a);
            input.set(1, 0, b);
            input.set(2, 0, c);
            input.set(3, 0, 1.0);
            pylidar::CMatrix<double> transOut = m_pMagneticMatrix->multiply(input);
            a = transOut.get(0, 0);
            b = transOut.get(1, 0);
            c = transOut.get(2, 0);
        }
        *pX = a;
        *pY = b;
        *pZ = c;
    }

private:
    Py_ssize_t m_nTotalPulsesReadFile;
    Py_ssize_t m_nPulsesToIgnore;
    pylidar::CVector<SRieglPulse> m_Pulses;
    pylidar::CVector<SRieglPoint> m_Points;
    npy_uint32 m_scanline;
    npy_uint16 m_scanlineIdx;
    pylidar::CMatrix<double> *m_pRotationMatrix;
    pylidar::CMatrix<double> *m_pMagneticMatrix;
};

// This class just reads the 'pose' parameters and is used by the
// _info function.
// This reads through the whole file and interepts relevant
// packets. If more than one packet has the info, then the values
// from the last one will be recorded at the end of the read.
class RieglParamReader : public scanlib::pointcloud
{
public:
    RieglParamReader() : scanlib::pointcloud(false),
        m_fLat(0), 
        m_fLong(0),
        m_fHeight(0),
        m_fHMSL(0),
        m_fRoll(NPY_NAN),
        m_fPitch(NPY_NAN),
        m_fYaw(NPY_NAN),
        m_beamDivergence(0),
        m_beamExitDiameter(0),
        m_thetaMin(0),
        m_thetaMax(0),
        m_phiMin(0),
        m_phiMax(0),
        m_thetaInc(0),
        m_phiInc(0),
        m_scanline(0),
        m_scanlineIdx(0),
        m_maxScanlineIdx(0),
        m_numPulses(0),
        m_bHaveData(false)
    {

    }

    // get all the information gathered in the read 
    // as a Python dictionary.
    PyObject *getInfoDictionary()
    {
        PyObject *pDict = PyDict_New();
        PyObject *pString, *pVal;

        // we assume that the values of these variables
        // (part of the pointcloud class itself) always exist
        // as they are probably part of the preamble so if any
        // reading of the stream has been done, they should be there.
        pVal = PyLong_FromLong(num_facets);
        PyDict_SetItemString(pDict, "NUM_FACETS", pVal);
        Py_DECREF(pVal);

        pVal = PyFloat_FromDouble(group_velocity);
        PyDict_SetItemString(pDict, "GROUP_VELOCITY", pVal);
        Py_DECREF(pVal);

        pVal = PyFloat_FromDouble(unambiguous_range);
        PyDict_SetItemString(pDict, "UNAMBIGUOUS_RANGE", pVal);
        Py_DECREF(pVal);
#if PY_MAJOR_VERSION >= 3
        pString = PyUnicode_FromString(serial.c_str());
#else
        pString = PyString_FromString(serial.c_str());
#endif
        PyDict_SetItemString(pDict, "SERIAL", pString);
        Py_DECREF(pString);
#if PY_MAJOR_VERSION >= 3
        pString = PyUnicode_FromString(type_id.c_str());
#else
        pString = PyString_FromString(type_id.c_str());
#endif
        PyDict_SetItemString(pDict, "TYPE_ID", pString);
        Py_DECREF(pString);
#if PY_MAJOR_VERSION >= 3
        pString = PyUnicode_FromString(build.c_str());
#else
        pString = PyString_FromString(build.c_str());
#endif
        PyDict_SetItemString(pDict, "BUILD", pString);
        Py_DECREF(pString);
        
        // now the fields that are valid if we have gathered 
        // from the 'pose' records
        if( m_bHaveData )
        {
            pVal = PyFloat_FromDouble(m_fLat);
            PyDict_SetItemString(pDict, "LATITUDE", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_fLong);
            PyDict_SetItemString(pDict, "LONGITUDE", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_fHeight);
            PyDict_SetItemString(pDict, "HEIGHT", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_fHMSL);
            PyDict_SetItemString(pDict, "HMSL", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_beamDivergence);
            PyDict_SetItemString(pDict, "BEAM_DIVERGENCE", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_beamExitDiameter);
            PyDict_SetItemString(pDict, "BEAM_EXIT_DIAMETER", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_thetaMin);
            PyDict_SetItemString(pDict, "THETA_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_thetaMax);
            PyDict_SetItemString(pDict, "THETA_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_phiMin);
            PyDict_SetItemString(pDict, "PHI_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_phiMax);
            PyDict_SetItemString(pDict, "PHI_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_thetaInc);
            PyDict_SetItemString(pDict, "THETA_INC", pVal);
            Py_DECREF(pVal);

            pVal = PyFloat_FromDouble(m_phiInc);
            PyDict_SetItemString(pDict, "PHI_INC", pVal);
            Py_DECREF(pVal);

            if( !npy_isnan(m_fRoll) )
            {
                pVal = PyFloat_FromDouble(m_fRoll);
                PyDict_SetItemString(pDict, "ROLL", pVal);
                Py_DECREF(pVal);
            }
            if( !npy_isnan(m_fPitch) )
            {
                pVal = PyFloat_FromDouble(m_fPitch);
                PyDict_SetItemString(pDict, "PITCH", pVal);
                Py_DECREF(pVal);
            }
            if( !npy_isnan(m_fYaw) )
            {
                pVal = PyFloat_FromDouble(m_fYaw);
                PyDict_SetItemString(pDict, "YAW", pVal);
                Py_DECREF(pVal);
            }

            if( !npy_isnan(m_fRoll) && !npy_isnan(m_fPitch) )
            {
                // now work out rotation matrix
                // pitch matrix
                pylidar::CMatrix<double> pitchMat(4, 4);
                pitchMat.set(0, 0, std::cos(m_fPitch));
                pitchMat.set(0, 1, 0.0);
                pitchMat.set(0, 2, std::sin(m_fPitch));
                pitchMat.set(0, 3, 0.0);
                pitchMat.set(1, 0, 0.0);
                pitchMat.set(1, 1, 1.0);
                pitchMat.set(1, 2, 0.0);
                pitchMat.set(1, 3, 0.0);
                pitchMat.set(2, 0, -std::sin(m_fPitch));
                pitchMat.set(2, 1, 0.0);
                pitchMat.set(2, 2, std::cos(m_fPitch));
                pitchMat.set(2, 3, 0.0);
                pitchMat.set(3, 0, 0.0);
                pitchMat.set(3, 1, 0.0);
                pitchMat.set(3, 2, 0.0);
                pitchMat.set(3, 3, 1.0);
            
                // roll matrix
                pylidar::CMatrix<double> rollMat(4, 4);
                rollMat.set(0, 0, 1.0);
                rollMat.set(0, 1, 0.0);
                rollMat.set(0, 2, 0.0);
                rollMat.set(0, 3, 0.0);
                rollMat.set(1, 0, 0.0);
                rollMat.set(1, 1, std::cos(m_fRoll));
                rollMat.set(1, 2, -std::sin(m_fRoll));
                rollMat.set(1, 3, 0.0);
                rollMat.set(2, 0, 0.0);
                rollMat.set(2, 1, std::sin(m_fRoll));
                rollMat.set(2, 2, std::cos(m_fRoll));
                rollMat.set(2, 3, 0.0);
                rollMat.set(3, 0, 0.0);
                rollMat.set(3, 1, 0.0);
                rollMat.set(3, 2, 0.0);
                rollMat.set(3, 3, 1.0);
            
                // yaw matrix; compass reading has been set to zero if nan
                pylidar::CMatrix<double> yawMat(4, 4);
                yawMat.set(0, 0, std::cos(m_fYaw));
                yawMat.set(0, 1, -std::sin(m_fYaw));
                yawMat.set(0, 2, 0.0);
                yawMat.set(0, 3, 0.0);
                yawMat.set(1, 0, std::sin(m_fYaw));
                yawMat.set(1, 1, std::cos(m_fYaw));
                yawMat.set(1, 2, 0.0);
                yawMat.set(1, 3, 0.0);
                yawMat.set(2, 0, 0.0);
                yawMat.set(2, 1, 0.0);
                yawMat.set(2, 2, 1.0);
                yawMat.set(2, 3, 0.0);
                yawMat.set(3, 0, 0.0);
                yawMat.set(3, 1, 0.0);
                yawMat.set(3, 2, 0.0);
                yawMat.set(3, 3, 1.0);

                // construct rotation matrix
                pylidar::CMatrix<double> tempMat = yawMat.multiply(pitchMat);
                pylidar::CMatrix<double> rotMat = tempMat.multiply(rollMat);

                pVal = (PyObject*)rotMat.getAsNumpyArray(NPY_DOUBLE);
                PyDict_SetItemString(pDict, "ROTATION_MATRIX", pVal);
                Py_DECREF(pVal);
            }

            // scanline info useful for building spatial index
            pVal = PyLong_FromLong(0);
            PyDict_SetItemString(pDict, "SCANLINE_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(m_scanline);
            PyDict_SetItemString(pDict, "SCANLINE_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(0);
            PyDict_SetItemString(pDict, "SCANLINE_IDX_MIN", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(m_maxScanlineIdx);
            PyDict_SetItemString(pDict, "SCANLINE_IDX_MAX", pVal);
            Py_DECREF(pVal);

            pVal = PyLong_FromLong(m_numPulses);
            PyDict_SetItemString(pDict, "NUMBER_OF_PULSES", pVal);
            Py_DECREF(pVal);
        }
        return pDict;
    }
protected:
    // Not sure what the difference between the functions below
    // is but they all have more or less the same data.
    // Get scanner position and orientation packet
    void on_scanner_pose_hr_1(const scanlib::scanner_pose_hr_1<iterator_type>& arg) 
    {
        scanlib::pointcloud::on_scanner_pose_hr_1(arg);
        m_bHaveData = true;
        m_fLat = arg.LAT;
        m_fLong = arg.LON;
        m_fHeight = arg.HEIGHT;
        m_fHMSL = arg.HMSL;
        if( !npy_isnan(arg.roll))
            m_fRoll = arg.roll * pi / 180.0;
        if( !npy_isnan(arg.pitch))
            m_fPitch = arg.pitch * pi / 180.0;
        if( !npy_isnan(arg.yaw))
            m_fYaw = arg.yaw * pi / 180.0;
        else
            m_fYaw = 0; // same as original code. Correct??
    }

    void on_scanner_pose_hr(const scanlib::scanner_pose_hr<iterator_type>& arg)
    {
        scanlib::pointcloud::on_scanner_pose_hr(arg);
        m_bHaveData = true;
        m_fLat = arg.LAT;
        m_fLong = arg.LON;
        m_fHeight = arg.HEIGHT;
        m_fHMSL = arg.HMSL;
        if( !npy_isnan(arg.roll))
            m_fRoll = arg.roll * pi / 180.0;
        if( !npy_isnan(arg.pitch))
            m_fPitch = arg.pitch * pi / 180.0;
        if( !npy_isnan(arg.yaw))
            m_fYaw = arg.yaw * pi / 180.0;
        else
            m_fYaw = 0; // same as original code. Correct??
    }

    void on_scanner_pose(const scanlib::scanner_pose<iterator_type>& arg)
    {
        scanlib::pointcloud::on_scanner_pose(arg);
        m_bHaveData = true;
        m_fLat = arg.LAT;
        m_fLong = arg.LON;
        m_fHeight = arg.HEIGHT;
        m_fHMSL = arg.HMSL;
        if( !npy_isnan(arg.roll))
            m_fRoll = arg.roll * pi / 180.0;
        if( !npy_isnan(arg.pitch))
            m_fPitch = arg.pitch * pi / 180.0;
        if( !npy_isnan(arg.yaw))
            m_fYaw = arg.yaw * pi / 180.0;
        else
            m_fYaw = 0; // same as original code. Correct??
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

    void on_shot()
    {
        m_scanlineIdx++;
        if( m_scanlineIdx > m_maxScanlineIdx )
        {
            m_maxScanlineIdx = m_scanlineIdx;
        }
        m_numPulses++;
    }

    // beam geometry
    void on_beam_geometry(const scanlib::beam_geometry<iterator_type>& arg) {
        scanlib::pointcloud::on_beam_geometry(arg);
        m_beamDivergence = arg.beam_divergence;
        m_beamExitDiameter = arg.beam_exit_diameter;
    }

    // scan configuration
    void on_scan_rect_fov(const scanlib::scan_rect_fov<iterator_type>& arg) {
        scanlib::pointcloud::on_scan_rect_fov(arg);
        m_thetaMin = arg.theta_min;
        m_thetaMax = arg.theta_max;
        m_phiMin = arg.phi_min;
        m_phiMax = arg.phi_max;
        m_phiInc = arg.phi_incr;
        m_thetaInc = arg.theta_incr;        
    }


private:
    double m_fLat;
    double m_fLong;
    double m_fHeight;
    double m_fHMSL;
    double m_fRoll;
    double m_fPitch;
    double m_fYaw;
    double m_beamDivergence;
    double m_beamExitDiameter;
    double m_thetaMin;
    double m_thetaMax;
    double m_phiMin;
    double m_phiMax;
    double m_thetaInc;
    double m_phiInc;
    long m_scanline;
    long m_scanlineIdx;
    long m_maxScanlineIdx;
    long m_numPulses;
    bool m_bHaveData;
};

PyObject *readWaveforms(fwifc_file waveHandle, fwifc_float64_t wave_v_group, 
    Py_ssize_t nPulseStart, Py_ssize_t nPulseEnd);

/* Python object wrapping a scanlib::basic_rconnection */
typedef struct
{
    PyObject_HEAD
    char *pszFilename; // so we can re-create the file obj if needed
    std::shared_ptr<scanlib::basic_rconnection> rc;
    scanlib::decoder_rxpmarker *pDecoder;
    scanlib::buffer *pBuffer;
    RieglReader *pReader;
    bool bFinishedReading;
    pylidar::CMatrix<double> *pMagneticDeclination;
    pylidar::CMatrix<double> *pRotationMatrix;

    // for waveforms, if present
    fwifc_file waveHandle;
    fwifc_float64_t wave_v_group; // group velocity
    fwifc_uint32_t wave_number_of_records; // assume the same as number of pulses

    // cached waveforms
    PyObject *pCachedWaveform;
    Py_ssize_t nCacheWavePulseStart;
    Py_ssize_t nCacheWavePulseEnd;

} PyRieglScanFile;

// return a dictionary with info about the file.
// means reading through the whole file
static PyObject *riegl_getFileInfo(PyObject *self, PyObject *args)
{
const char *pszFilename;

    if( !PyArg_ParseTuple(args, "s", &pszFilename) )
        return NULL;

    RieglParamReader reader;
    try
    {
        std::shared_ptr<scanlib::basic_rconnection> rc = scanlib::basic_rconnection::create(pszFilename);

        scanlib::decoder_rxpmarker dec(rc);

        scanlib::buffer buf;

        for(dec.get(buf); !dec.eoi(); dec.get(buf))
        {
            reader.dispatch(buf.begin(), buf.end());
        }
    }
    catch(scanlib::scanlib_exception e)
    {
        // raise Python exception
        PyErr_Format(GETSTATE(self)->error, "Error from Riegl lib: %s", e.what());
        return NULL;
    }   

    PyObject *pDict = reader.getInfoDictionary();
    return pDict;
}

static const char *SupportedDriverOptions[] = {"ROTATION_MATRIX", "MAGNETIC_DECLINATION", NULL};
static PyObject *riegl_getSupportedOptions(PyObject *self, PyObject *args)
{
    return pylidar_stringArrayToTuple(SupportedDriverOptions);
}

// module methods
static PyMethodDef module_methods[] = {
    {"getFileInfo", (PyCFunction)riegl_getFileInfo, METH_VARARGS,
        "Get a dictionary with information about the file. Pass the filename"},
    {"getSupportedOptions", (PyCFunction)riegl_getSupportedOptions, METH_NOARGS,
        "Get a tuple of supported driver options"},
    {NULL}  /* Sentinel */
};

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
        module_methods,
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
    free(self->pszFilename);
    self->rc->close();
    self->rc.reset();
    if( self->waveHandle != NULL )
    {
        fwifc_close(self->waveHandle);
    }
    //fprintf(stderr, "total read %ld\n", self->pReader->getNumPulsesReadFile());
    delete self->pDecoder;
    delete self->pBuffer;
    delete self->pReader;
    Py_XDECREF(self->pCachedWaveform);
    if( self->pMagneticDeclination != NULL)
        delete self->pMagneticDeclination;
    if( self->pRotationMatrix != NULL )
        delete self->pRotationMatrix;

    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Set a Python exception from wave errorcode */
void setWaveError(fwifc_int32_t result)
{
    fwifc_csz message;
    fwifc_get_last_error(&message);
    // raise Python exception
    PyErr_Format(GETSTATE_FC->error, "Error from Riegl wave lib: %s", message);
}

/* init method - open file */
/* takes rxp filename, wfm filename (can be NULL) and dictionary of options */
/* currently only the SupportedDriverOptions keys are used */

static int 
PyRieglScanFile_init(PyRieglScanFile *self, PyObject *args, PyObject *kwds)
{
char *pszFname = NULL, *pszWaveFname;
PyObject *pOptionDict;

    if( !PyArg_ParseTuple(args, "szO", &pszFname, &pszWaveFname, &pOptionDict ) )
    {
        return -1;
    }

    if( !PyDict_Check(pOptionDict) )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Last parameter to init function must be a dictionary");
        return -1;
    }

    // Deal with reading options
    // "ROTATION_MATRIX", "MAGNETIC_DECLINATION"
    PyObject *pRotationMatrix = PyDict_GetItemString(pOptionDict, "ROTATION_MATRIX");
    if( pRotationMatrix != NULL )
    {
        if( !PyArray_Check(pRotationMatrix) || (PyArray_NDIM((PyArrayObject*)pRotationMatrix) != 2) ||
            (PyArray_DIM((PyArrayObject*)pRotationMatrix, 0) != 4 ) || (PyArray_DIM((PyArrayObject*)pRotationMatrix, 1) != 4 ) )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "ROTATION_MATRIX must be a 4x4 numpy array");    
            return -1;
        }

        // ensure Float64
        PyArrayObject *pRotationMatrixF64 = (PyArrayObject*)PyArray_FROM_OT(pRotationMatrix, NPY_FLOAT64);
        // make our matrix
        try
        {
            self->pRotationMatrix = new pylidar::CMatrix<double>(pRotationMatrixF64);
        }
        catch(std::exception e)
        {
            Py_DECREF(pRotationMatrixF64);
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, e.what());
            return -1;
        }

        Py_DECREF(pRotationMatrixF64);
    }
    else
    {
        self->pRotationMatrix = NULL;
    }

    PyObject *pMagneticDeclination = PyDict_GetItemString(pOptionDict, "MAGNETIC_DECLINATION");
    if( pMagneticDeclination != NULL )
    {
        if( !PyFloat_Check(pMagneticDeclination) )
        {
            // raise Python exception
            PyErr_SetString(GETSTATE_FC->error, "MAGNETIC_DECLINATION must be a floating point number");
            return -1;
        }

        double dMagneticDeclination = PyFloat_AsDouble(pMagneticDeclination);
        double mdrad = dMagneticDeclination * NPY_PI / 180.0;

        self->pMagneticDeclination = new pylidar::CMatrix<double>(4, 4);
        self->pMagneticDeclination->set(0, 0, std::cos(mdrad));
        self->pMagneticDeclination->set(0, 1, -std::sin(mdrad));
        self->pMagneticDeclination->set(0, 2, 0.0);
        self->pMagneticDeclination->set(0, 3, 0.0);
        self->pMagneticDeclination->set(1, 0, std::sin(mdrad));
        self->pMagneticDeclination->set(1, 1, std::cos(mdrad));
        self->pMagneticDeclination->set(1, 2, 0.0);
        self->pMagneticDeclination->set(1, 3, 0.0);
        self->pMagneticDeclination->set(2, 0, 0.0);
        self->pMagneticDeclination->set(2, 1, 0.0);
        self->pMagneticDeclination->set(2, 2, 1.0);
        self->pMagneticDeclination->set(2, 3, 0.0);
        self->pMagneticDeclination->set(3, 0, 0.0);
        self->pMagneticDeclination->set(3, 1, 0.0);
        self->pMagneticDeclination->set(3, 2, 0.0);
        self->pMagneticDeclination->set(3, 3, 1.0);
    }
    else
    {    
        self->pMagneticDeclination = NULL;
    }

    try
    {
        // take a copy of the filename so we can re
        // create the file pointer.
        self->pszFilename = strdup(pszFname);

        self->rc = scanlib::basic_rconnection::create(pszFname);

        // The decoder class scans off distinct packets from the continuous data stream
        // i.e. the rxp format and manages the packets in a buffer.
        self->pDecoder = new scanlib::decoder_rxpmarker(self->rc);

        // The buffer is a structure that holds pointers into the decoder buffer
        // thereby avoiding unnecessary copies of the data.
        self->pBuffer = new scanlib::buffer();

        // our reader class
        self->pReader = new RieglReader(self->pRotationMatrix, self->pMagneticDeclination);
    }
    catch(scanlib::scanlib_exception e)
    {
        // raise Python exception
        PyErr_Format(GETSTATE_FC->error, "Error from Riegl lib: %s", e.what());
        return -1;
    }   

    self->pCachedWaveform = NULL;
    self->nCacheWavePulseStart = 0;
    self->nCacheWavePulseEnd = 0;


    if( pszWaveFname != NULL )
    {
        // first check that the version of the lib matches the compile time version
        fwifc_uint16_t api_major, api_minor;
        fwifc_csz build_version, build_tag;
        fwifc_get_library_version(&api_major, &api_minor, &build_version, &build_tag);
        if( api_major != RIEGL_WFM_MAJOR )
        {
            // raise Python exception
            PyErr_Format(GETSTATE_FC->error, "Mismatched libraries - Riegl waveform lib differs in major version number. "
                "Was compiled against version %d.%d. Now running with %d.%d\n", 
                RIEGL_WFM_MAJOR, RIEGL_WFM_MINOR, (int)api_major, (int)api_minor);
            return -1;
        }

        // waveforms are present. Open the file
        fwifc_int32_t result = fwifc_open(pszWaveFname, &self->waveHandle);
        if(result != 0)
        {
            setWaveError(result);
            return -1;
        }

        // get the info. We are only really interested in the velocity
        // so we save that to self
        fwifc_csz instrument;           /* the instrument type */
        fwifc_csz serial;               /* serial number of the instrument */
        fwifc_csz epoch;                /* of time_external can be a datetime */
                                        /* "2010-11-16T00:00:00" or */
                                        /* "DAYSEC" or "WEEKSEC" or */
                                        /* "UNKNOWN" if not known */
        fwifc_float64_t sampling_time;  /* sampling interval in seconds */
        fwifc_uint16_t flags;           /* GPS synchronized, ... */
        fwifc_uint16_t num_facets;      /* number of mirror facets */
        result = fwifc_get_info(self->waveHandle, 
                    &instrument,
                    &serial,
                    &epoch,
                    &self->wave_v_group,
                    &sampling_time,
                    &flags,
                    &num_facets);
        if(result != 0)
        {
            setWaveError(result);
            return -1;
        }

        // set time to relative
        result = fwifc_set_sosbl_relative(self->waveHandle, 1);
        if(result != 0)
        {
            setWaveError(result);
            return -1;
        }

        // get the number of records - can assume this is the same
        // as the number of pulses - I think.
        fwifc_uint32_t currposs;
        fwifc_tell(self->waveHandle, &currposs);
        fwifc_seek(self->waveHandle, 0xFFFFFFFF);
        fwifc_tell(self->waveHandle, &self->wave_number_of_records);
        fwifc_seek(self->waveHandle, currposs);
        //fprintf(stderr, "number of wave records %d\n", self->wave_number_of_records);*/
    }
    else
    {
        self->waveHandle = NULL;
    }

    self->bFinishedReading = false;
    return 0;
}

static PyObject *PyRieglScanFile_readData(PyRieglScanFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd, nPulses;
    if( !PyArg_ParseTuple(args, "nn:readData", &nPulseStart, &nPulseEnd ) )
        return NULL;

    nPulses = nPulseEnd - nPulseStart;

    Py_ssize_t nPulsesToIgnore = 0; // how many to ignore before doing stuff
    // get number read minus what is in the buffer - where we are 'officially' up to
    Py_ssize_t nTotalRead = self->pReader->getNumPulsesReadFile() - self->pReader->getNumPulsesRead();
    if( nPulseStart < nTotalRead )
    {
        //fprintf(stderr, "going back to start\n");
        // need to read earlier stuff in the file. 
        // reset to beginning and start looping
        // I couldn't get the seekg call to work so 
        // re create basic_rconnection instead.
        self->rc.reset();
        self->rc = scanlib::basic_rconnection::create(self->pszFilename);
        // ensure buffers are flushed
        delete self->pDecoder;
        delete self->pBuffer;
        delete self->pReader;
        self->pDecoder = new scanlib::decoder_rxpmarker(self->rc);
        self->pBuffer = new scanlib::buffer();
        self->pReader = new RieglReader(self->pRotationMatrix, self->pMagneticDeclination);
        self->pReader->setPulsesToIgnore(nPulseStart);
    }
    else if( nPulseStart > nTotalRead )
    {
        // requested range is after current location
        nPulsesToIgnore = nPulseStart - self->pReader->getNumPulsesReadFile();
        //fprintf(stderr, "missing values %ld\n", nPulsesToIgnore);
        self->pReader->setPulsesToIgnore(nPulsesToIgnore);
        // move the extra data down by that many too
        // so we can use any still in the buffer that we will need
        self->pReader->removeLowerPulses(nPulsesToIgnore);
    }
    else
    {
        self->pReader->setPulsesToIgnore(0);
    }
    // there may be stuff in the reader's buffer already but that should
    // be ok since we have handled making it ok above

    // loop through the requested number of pulses - always 
    // do one more so we get all the points (which are normally after
    // each pulse).
    if( self->pReader->getNumPulsesRead() < (nPulses+1) ) // don't bother if we already have enough from last time
    {
        try
        {
            while(!self->pDecoder->eoi() && (self->pReader->getNumPulsesRead() < (nPulses+1)))
            {
                self->pDecoder->get(*self->pBuffer); 
                self->pReader->dispatch(self->pBuffer->begin(), self->pBuffer->end());
            }
        }
        catch(scanlib::scanlib_exception e)
        {
            // raise Python exception
            PyErr_Format(GETSTATE_FC->error, "Error from Riegl lib: %s", e.what());
            return NULL;
        }
    }

    if( self->waveHandle != NULL )
    {
        // ok this is a bit hairy. We need to get the wfm_start_idx and number_of_waveform_samples fields
        // but we can only get them by reading the waveforms. So we read the waveforms, then try to save
        // a bit of time by caching them in case they are asked for/provided by _readWaveforms.
        if( ( self->nCacheWavePulseStart != nPulseStart ) || ( self->nCacheWavePulseEnd != nPulseEnd) )
        {
            self->pCachedWaveform = readWaveforms(self->waveHandle, self->wave_v_group, nPulseStart, nPulseEnd);
            self->nCacheWavePulseStart = nPulseStart;
            self->nCacheWavePulseEnd = nPulseEnd;
        }
        PyArrayObject *wfmStart = (PyArrayObject*)PyTuple_GetItem(self->pCachedWaveform, 2);
        PyArrayObject *wfmCount = (PyArrayObject*)PyTuple_GetItem(self->pCachedWaveform, 3);
        for( npy_intp n = 0; n < PyArray_DIM(wfmStart, 0); n++)
        {
            npy_uint32 st = *(npy_uint32*)PyArray_GETPTR1(wfmStart, n);
            npy_uint8 ct = *(npy_uint8*)PyArray_GETPTR1(wfmCount, n);
            self->pReader->setWaveformInfo(n, st, ct);
        }
    }

    // get pulse array as numpy array
    Py_ssize_t point_idx;
    PyArrayObject *pPulses = self->pReader->getPulses(nPulses, &point_idx); 
    // points
    PyArrayObject *pPoints = self->pReader->getPoints(point_idx);

    // we have finished if we are at the end
    self->bFinishedReading = self->pDecoder->eoi() && (self->pReader->getNumPulsesRead() == 0);
    //fprintf(stderr, "n = %ld\n", self->pReader->getNumPulsesReadFile());

    // build tuple
    PyObject *pTuple = PyTuple_Pack(2, pPulses, pPoints);
    Py_DECREF(pPulses);
    Py_DECREF(pPoints);

    return pTuple;
}

// returns a tuple with waveform info, received, wfmStart (for pulses) and wfmCount (for pulses)
static PyObject *PyRieglScanFile_readWaveforms(PyRieglScanFile *self, PyObject *args)
{
    Py_ssize_t nPulseStart, nPulseEnd;
    if( !PyArg_ParseTuple(args, "nn:readWaveforms", &nPulseStart, &nPulseEnd ) )
        return NULL;

    if( self->waveHandle == NULL )
    {
        // raise Python exception
        PyErr_SetString(GETSTATE_FC->error, "Waveform file not present (or passed to constructor)");
        return NULL;
    }

    // PyRieglScanFile_readData also reads the waveforms to find the wfm_start_idx and number_of_waveform_samples
    // and caches all the data so we can return it if the range is the same.
    if( ( self->nCacheWavePulseStart != nPulseStart ) || ( self->nCacheWavePulseEnd != nPulseEnd) )
    {
        //fprintf(stderr, "new waveform\n");
        self->pCachedWaveform = readWaveforms(self->waveHandle, self->wave_v_group, nPulseStart, nPulseEnd);
        self->nCacheWavePulseStart = nPulseStart;
        self->nCacheWavePulseEnd = nPulseEnd;
    }
    //else
    //   fprintf(stderr, "returning cached wfm\n");

    Py_INCREF(self->pCachedWaveform);
    return self->pCachedWaveform;
}

// Does the actual reading of the waveforms. 
// returns a tuple with waveform info, received, wfmStart (for pulses) and wfmCount (for pulses)
PyObject *readWaveforms(fwifc_file waveHandle, fwifc_float64_t wave_v_group, 
        Py_ssize_t nPulseStart, Py_ssize_t nPulseEnd)
{
    pylidar::CVector<SRieglWaveformInfo> waveInfo(nInitSize, nGrowBy);
    pylidar::CVector<npy_uint16> received(nInitSize, nGrowBy);
    pylidar::CVector<npy_uint32> wfmStart(nInitSize, nGrowBy);
    pylidar::CVector<npy_uint8> wfmNumber(nInitSize, nGrowBy);

    Py_ssize_t nPulses = nPulseEnd - nPulseStart;

    fwifc_float64_t time_sorg;      /* start of range gate in s */
    fwifc_float64_t time_external;  /* external time in s relative to epoch */
    fwifc_float64_t origin[3];      /* origin vector in m */
    fwifc_float64_t direction[3];   /* direction vector (dimensionless) */
    fwifc_uint16_t  facet;          /* facet number (0 to num_facets-1) */
    fwifc_uint32_t  sbl_count;      /* number of sample blocks */
    fwifc_uint32_t  sbl_size;       /* size of sample block in bytes */
    fwifc_sbl_t*    psbl_first;     /* pointer to first sample block */
    fwifc_float64_t time_ref = 0;       /* emission time in s */
    fwifc_uint16_t flags;           /* GPS synchronized, ... */

    npy_uint64 nRecStartIdx = 0;
    SRieglWaveformInfo info;
    npy_uint16 waveSample;
    npy_uint32 waveformStartIdx = 0;
    npy_uint8 waveformCount;

    // seek to the first pulse
    // conveninently you can seek with the wave lib
    fwifc_int32_t result = fwifc_seek(waveHandle, nPulseStart+1); // starts at record 1
    if( result != 0 )
    {
        setWaveError(result);
        return NULL;
    }

    for( Py_ssize_t n = 0; n < nPulses; n++ )
    {
        result = fwifc_read(waveHandle,
                            &time_sorg,
                            &time_external,
                            &origin[0],
                            &direction[0],
                            &flags,
                            &facet,
                            &sbl_count,
                            &sbl_size,
                            &psbl_first);
        if(result == FWIFC_END_OF_FILE)
            break;
        else if(result != 0)
        {
            setWaveError(result);
            return NULL;
        }

        // go through each of the samples
        fwifc_sbl_t* psbl = psbl_first;
        waveformCount = 0;
        for (fwifc_uint32_t sbi = 0; sbi < sbl_count; ++sbi)
        {
            if( psbl->channel == 3 ) // Reference time
            {
                // the reference channel represents the emitted (reference)
                // signal. This signal is not directly available for V-Line scanners.
                // Instead, these scanners provide an accurate value of the timestamp 
                // of laser pulse emission. For all other scanners tref has to be 
                // determined from the reference channel waveform.
                
                time_ref = psbl->time_sosbl;
            }
            else if( ( psbl->channel == 0 ) || ( psbl->channel == 1 ) )
            {
                
                // high power channel (0) and the more sensitive 
                // low power channel (1) are read
                // the saturation channel (2) is currently ignored
                
                // first the info
                info.number_of_waveform_received_bins = psbl->sample_count;
                info.range_to_waveform_start = (psbl->time_sosbl - time_ref) * 
                                                (wave_v_group / 2.0);
                info.received_start_idx = nRecStartIdx;
                info.channel = psbl->channel;
                // Riegl gives no offset/scaling 
                info.receive_wave_gain = 1.0;
                info.receive_wave_offset = 0.0;
                waveInfo.push(&info);
                waveformCount++;

                // now the actual waveform
                for(fwifc_uint32_t nsample = 0; nsample < psbl->sample_count; nsample++)
                {
                    waveSample = psbl->sample[nsample];
                    received.push(&waveSample);
                }

                nRecStartIdx += psbl->sample_count;
            }
            psbl++;
        }

        // update the info for the pulses
        wfmStart.push(&waveformStartIdx);
        wfmNumber.push(&waveformCount);
        waveformStartIdx++;
    }

    // extract values as numpy arrays
    PyArrayObject *pNumpyInfo = waveInfo.getNumpyArray(RieglWaveformInfoFields);
    PyArrayObject *pNumpyRec = received.getNumpyArray(NPY_UINT16);
    PyArrayObject *pNumpyWfmStart = wfmStart.getNumpyArray(NPY_UINT32);
    PyArrayObject *pNumpyWfmNumber = wfmNumber.getNumpyArray(NPY_UINT8);

    // build tuple
    PyObject *pTuple = PyTuple_Pack(4, pNumpyInfo, pNumpyRec, pNumpyWfmStart, pNumpyWfmNumber);
    Py_DECREF(pNumpyInfo);
    Py_DECREF(pNumpyRec);
    Py_DECREF(pNumpyWfmStart);
    Py_DECREF(pNumpyWfmNumber);

    return pTuple;
}

/* Table of methods */
static PyMethodDef PyRieglScanFile_methods[] = {
    {"readData", (PyCFunction)PyRieglScanFile_readData, METH_VARARGS, NULL},
    {"readWaveforms", (PyCFunction)PyRieglScanFile_readWaveforms, METH_VARARGS, NULL},
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
    return PyLong_FromSsize_t(self->pReader->getNumPulsesReadFile());
}

static PyObject *PyRieglScanFile_getNumWaveRecords(PyRieglScanFile *self, void *closure)
{
    if( self->waveHandle != NULL )
        return PyLong_FromLong(self->wave_number_of_records);
    else
        Py_RETURN_NONE;
}

/* get/set */
static PyGetSetDef PyRieglScanFile_getseters[] = {
    {(char*)"finished", (getter)PyRieglScanFile_getFinished, NULL, (char*)"Get Finished reading state", NULL}, 
    {(char*)"pulsesRead", (getter)PyRieglScanFile_getPulsesRead, NULL, (char*)"Get number of pulses read", NULL},
    {(char*)"numWaveRecords", (getter)PyRieglScanFile_getNumWaveRecords, NULL, (char*)"Get number of waveform records in file", NULL},
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
    pModule = Py_InitModule("_riegl", module_methods);
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
    PyModule_AddObject(pModule, "error", state->error);

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
