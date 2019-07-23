"""
Driver for GEDI L1A01 HDF5 files.

Read Driver Options
-------------------

These are contained in the READSUPPORTEDOPTIONS module level variable.

+-----------------------+--------------------------------------------+
| Name                  | Use                                        |
+=======================+============================================+
| POINT_FROM            | A 3 element tuple defining which fields to |
|                       | create a fake point from (x,y,z). Default  |
|                       | is ('GEO_longitude_lastbin',               |
|                       |     'GEO_latitude_lastbin',                |
|                       |     'GEO_elevation_lastbin')               |
+-----------------------+--------------------------------------------+
| BEAM                  | A string defining the beam id to use. One  |
|                       | of ['BEAM0000', 'BEAM0001', 'BEAM0010',    |
|                       | 'BEAM0011', 'BEAM0101', 'BEAM0110',        |
|                       | 'BEAM1000', 'BEAM1011']                    |
+-----------------------+--------------------------------------------+
| PULSE_GROUP_NAMES     | A list or tuple defining the groups with   |
|                       | per pulse data to use.                     |
|                       | Can be any of ['CLK', 'INST_HK',           |
|                       | 'TX_PROCESSING', 'geolocation',            |
|                       | 'geophys_corr']                          |
+-----------------------+--------------------------------------------+

Write Driver Options
--------------------

These are contained in the WRITESUPPORTEDOPTIONS module level variable.

+-----------------------+--------------------------------------------+
| Name                  | Use                                        |
+=======================+============================================+
| HDF5_CHUNK_SIZE       | Set the HDF5 chunk size when creating      |
|                       | columns. Defaults to 250.                  |
+-----------------------+--------------------------------------------+

"""

# This file is part of PyLidar
# Copyright (C) 2019 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import print_function, division

import sys
import h5py
import copy
import numpy
from numba import jit
from . import h5space
from . import generic
from . import gridindexutils

WRITESUPPORTEDOPTIONS = ('HDF5_CHUNK_SIZE')
"driver options"

"Default hdf5 chunk size set on column creation"
DEFAULT_HDF5_CHUNK_SIZE = 250

READSUPPORTEDOPTIONS = ('POINT_FROM', 'BEAM', 'PULSE_GROUP_NAMES')
"Supported read options"

WRITESUPPORTEDOPTIONS = ('BEAM', 'PULSE_GROUP_NAMES')
"Supported write options"

DEFAULT_POINT_FROM = ('GEO_longitude_lastbin', 'GEO_latitude_lastbin', 'GEO_elevation_lastbin')
DEFAULT_BEAM = 'BEAM0101'
DEFAULT_PULSE_GROUP_NAMES = ['CLK', 'INST_HK', 'TX_PROCESSING', 'geolocation', 'geophys_corr']
EXPECTED_HEADER_FIELDS = ['l0_to_l1a_githash', 'l0_to_l1a_version']
CLASSIFICATION_NAME = 'CLASSIFICATION'
"GEDI L1A01 files don't have a CLASSIFICATION column so we have to create a blank one"


MAIN_FIELDS = {'all_samples_sum': numpy.uint32, 'bdu': numpy.uint16, 'beam': numpy.uint16, 'channel': numpy.uint8, 
'dithered': numpy.uint8, 'framesync': numpy.uint32, 'gps_framesync': numpy.uint32, 'gps_time': numpy.uint32, 
'is_crc_valid': numpy.uint8, 'iss_fraction': numpy.uint16, 'iss_framesync': numpy.uint32, 'iss_time': numpy.uint32, 
'master_frac': numpy.float64, 'master_int': numpy.uint32, 'master_source': numpy.uint8, 'minor_frame_number': numpy.uint16, 
'noise_mean': numpy.float32, 'noise_std_dev': numpy.uint16, 'override': numpy.uint8, 'packet_count': numpy.uint32, 
'packet_length': numpy.uint16, 'pers_coarse': numpy.uint8, 'pers_fine': numpy.uint8, 'rx_offset': numpy.uint16, 
'rx_open': numpy.uint32, 'rx_sample': numpy.uint8, 'rx_sample_sum': numpy.uint32, 'selection_stretchers_x': numpy.uint16, 
'selection_stretchers_y': numpy.uint16, 'shot_number': numpy.uint64, 'stale_return_flag': numpy.uint8, 
'stretchers_x': numpy.uint8, 'stretchers_y': numpy.uint8, 'sum_of_squares': numpy.uint32, 'sync': numpy.uint16, 
'th_coarse_used': numpy.uint16, 'th_left_used': numpy.uint16, 'th_mult_coarse': numpy.float32, 'th_mult_left': numpy.float32, 
'th_mult_right': numpy.float32, 'th_right_used': numpy.uint16, 'tx_offset': numpy.uint16, 'tx_open': numpy.uint32, 
'tx_sample_info': numpy.uint8, 'tx_sample_start_index': numpy.uint64, 'tx_sample_sum': numpy.uint32}
"These fields have defined type"

CLK_FIELDS = {'dFF': numpy.float64, 'fit': numpy.float64, 'interp': numpy.uint8, 'range_bin0_m': numpy.float64, 'range_lastbin_m': numpy.float64}
"These fields have defined type"

INST_HK_FIELDS = {'adc_t': numpy.float64, 'adccal': numpy.float64, 'bgngate_frac': numpy.float64, 'bgngate_usec': numpy.float64, 'dma_det_t': numpy.float64, 
'dma_gain_mode': numpy.float64, 'dma_htr_i': numpy.float64, 'dma_hv_i': numpy.float64, 'dma_hvbias_cmd': numpy.float64, 'dma_hvbias_meas': numpy.float64, 
'dma_hvbias_setpt': numpy.float64, 'dma_n5p0_v': numpy.float64, 'dma_p12_v': numpy.float64, 'dma_p5p0_v': numpy.float64, 'dmb_det_t': numpy.float64, 
'dmb_gain_mode': numpy.float64, 'dmb_htr_i': numpy.float64, 'dmb_hv_i': numpy.float64, 'dmb_hvbias_cmd': numpy.float64, 'dmb_hvbias_meas': numpy.float64, 
'dmb_hvbias_setpt': numpy.float64, 'dmb_n5p0_v': numpy.float64, 'dmb_p12_v': numpy.float64, 'dmb_p5p0_v': numpy.float64, 'interp': numpy.uint8, 
'interp_even_odd_nsemean': numpy.uint8, 'last_lzr_mode': numpy.float64, 'lzr_mode': numpy.float64, 'nsemean': numpy.float64, 'nsemean_even': numpy.float64, 
'nsemean_odd': numpy.float64, 'rxgate_frac': numpy.float64, 'rxgate_usec': numpy.float64, 'rxgateadj_frac': numpy.float64, 'rxgateadj_usec': numpy.float64, 
'time_since_last_lzr_mode': numpy.float64, 'txgate_frac': numpy.float64, 'txgate_usec': numpy.float64}
"These fields have defined type"

TX_PROCESSING_FIELDS = {'cnttx': numpy.int64, 'mean_binset1': numpy.float64, 'mean_binset1_2': numpy.float64, 'mean_binset2': numpy.float64, 
'mean_binset2_2': numpy.float64, 'mean_binset3': numpy.float64, 'mean_binset3_2': numpy.float64, 'mean_binset4': numpy.float64, 'mean_binset4_2': numpy.float64, 
'mean_binset_even': numpy.float64, 'mean_binset_even_2': numpy.float64, 'mean_binset_odd': numpy.float64, 'mean_binset_odd_2': numpy.float64, 
'numbins_averaged': numpy.float64, 'range_bin0': numpy.float64, 'range_bin0_error': numpy.float64, 'range_lastbin': numpy.float64, 'range_lastbin_error': numpy.float64, 
'tx_ampflag': numpy.int32, 'tx_cntsat': numpy.int64, 'tx_egamplitude': numpy.float64, 'tx_egamplitude_error': numpy.float64, 
'tx_egbias': numpy.float64, 'tx_egbias_error': numpy.float64, 'tx_egcenter': numpy.float64, 'tx_egcenter_error': numpy.float64, 
'tx_egchisq': numpy.float64, 'tx_egflag': numpy.int32, 'tx_eggamma': numpy.float64, 'tx_eggamma_error': numpy.float64, 
'tx_egiters': numpy.int32, 'tx_egsigma': numpy.float64, 'tx_egsigma_error': numpy.float64, 'tx_energy': numpy.float64, 
'tx_gamplitude': numpy.float64, 'tx_gamplitude_error': numpy.float64, 'tx_gbias': numpy.float64, 'tx_gbias_error': numpy.float64, 
'tx_gchisq': numpy.float64, 'tx_gflag': numpy.int32, 'tx_giters': numpy.int32, 'tx_gloc': numpy.float64, 'tx_gloc_error': numpy.float64, 
'tx_gwidth': numpy.float64, 'tx_gwidth_error': numpy.float64, 'tx_maxamp': numpy.float64, 'tx_maxcorr': numpy.float64, 'tx_maxppcorr': numpy.float64, 
'tx_minamp': numpy.float64, 'tx_peakloc': numpy.int64, 'tx_pploc': numpy.float64, 'tx_pulseflag': numpy.int32, 'tx_ringflag': numpy.int32, 
'tx_satflag': numpy.int32, 'tx_sd_nw': numpy.float64, 'waveform_stack_count': numpy.int64}
"These fields have defined type"

TX_PROCESSING_SUMMARY_FIELDS = {'short_term_av_end_time_frac': numpy.float64, 'short_term_av_end_time_int': numpy.int64, 'short_term_av_start_time_frac': numpy.float64, 
'short_term_av_start_time_int': numpy.int64, 'short_term_av_tx_waveform': numpy.float64, 'rxwaveform': numpy.uint16, 'txwaveform': numpy.uint16}

ANCILLARY_FIELDS = {'master_time_epoch': numpy.float64}
"These fields have defined type"

GEOLOCATION_FIELDS = {'altitude_spacecraft': numpy.float64, 'bounce_time_offset_bin0': numpy.float64, 'bounce_time_offset_lastbin': numpy.float64, 
'delta_time': numpy.float64, 'digital_elevation_model': numpy.float64, 'elevation_bin0': numpy.float64, 'elevation_bin0_error': numpy.float64, 
'elevation_lastbin': numpy.float64, 'elevation_lastbin_error': numpy.float64, 'latitude_bin0': numpy.float64, 'latitude_lastbin': numpy.float64, 
'latitude_spacecraft': numpy.float64, 'local_beam_azimuth': numpy.float64, 'local_beam_elevation': numpy.float64, 'longitude_bin0': numpy.float64, 
'longitude_lastbin': numpy.float64, 'longitude_spacecraft': numpy.float64, 'mean_sea_surface': numpy.float64, 'neutat_delay_derivative_bin0': numpy.float64, 
'neutat_delay_derivative_lastbin': numpy.float64, 'neutat_delay_total_bin0': numpy.float64, 'neutat_delay_total_lastbin': numpy.float64, 
'range_bias_correction': numpy.float64, 'shot_number': numpy.float64, 'solar_azimuth': numpy.float64, 'solar_elevation': numpy.float64, 
'surface_type': numpy.int8}
"These fields have defined type"

GEOPHYS_CORR_FIELDS = {'delta_time': numpy.float64, 'dynamic_atmosphere_correction': numpy.float64, 'geoid': numpy.float64, 
'tide_earth': numpy.float64, 'tide_load': numpy.float64, 'tide_ocean': numpy.float64, 'tide_ocean_pole': numpy.float64, 
'tide_pole': numpy.float64}
"These fields have defined type"

WAVEFORMINFO_FIELDS = {'tx_sample_count': numpy.uint16, 'tx_sample_start_index': numpy.uint64, 'rx_sample_count': numpy.uint16, 
'rx_sample_start_index': numpy.uint64}

GROUP_IDS = {'CLK': 'CLK', 'INST_HK': 'IHK', 'TX_PROCESSING': 'TXP', 'ancillary': 'ANC', 'geolocation': 'GEO', 'geophys_corr': 'GCO'}
GROUP_NAMES = {'CLK': 'CLK', 'IHK': 'INST_HK', 'TXP': 'TX_PROCESSING', 'ANC': 'ancillary', 'GEO': 'geolocation', 'GCO': 'geophys_corr'}


@jit
def flatten2dWaveformData(wavedata, inmask, nrecv, flattened):
    """
    Helper routine that flattens transmitted or received into something that can be written. 
    wavedata is either the (2d) transmitted or received
    inmask is the .mask for wavedata
    nrecv is the output array for rx_sample_count etc
    flattened is the flattened version of wavedata
    """
    nsamples = wavedata.shape[0]
    npulses = wavedata.shape[1]
    flat_idx = 0
    for p in range(npulses):
        for s in range(nsamples):
            if not inmask[s, p]:
                flattened[flat_idx] = wavedata[s, p]
                flat_idx += 1
                nrecv[p] += 1

    def flatten2dMaskedArray(flatArray, in2d, mask2d, idx2d):
    """
    Used by writeData to flatten out masked 2d data into a 1d
    using the indexes and masked saved from when the array was created.
    """
    (maxPts, nRows) = in2d.shape
    for n in range(maxPts):
        for row in range(nRows):
            if not mask2d[n, row]:
                idx = idx2d[n, row]
                val = in2d[n, row]
                flatArray[idx] = val

class GEDIL1A01File(generic.LiDARFile):
    """
    Reader for GEDI L1A01 files
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)    

        # convert mode into h5py mode string
        if mode == generic.READ:
            h5py_mode = 'r'
        elif mode == generic.UPDATE:
            h5py_mode = 'r+'
        elif mode == generic.CREATE:
            h5py_mode = 'w'
        else:
            raise ValueError('Unknown value for mode parameter')

        # check driver options    
        if mode == generic.READ:
            options = READSUPPORTEDOPTIONS
        else:
            options = WRITESUPPORTEDOPTIONS
        for key in userClass.lidarDriverOptions:
            if key not in options:
                msg = '%s not a supported GEDIL1A01 option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)

        # set driver options
        self.pointFrom = DEFAULT_POINT_FROM
        if 'POINT_FROM' in userClass.lidarDriverOptions:
            self.pointFrom = userClass.lidarDriverOptions['POINT_FROM']
        self.beam = DEFAULT_BEAM
        if 'BEAM' in userClass.lidarDriverOptions:
            self.beam = userClass.lidarDriverOptions['BEAM']
        self.pulse_group_names = DEFAULT_PULSE_GROUP_NAMES
        if 'PULSE_GROUP_NAMES' in userClass.lidarDriverOptions:
            self.pulse_group_names = userClass.lidarDriverOptions['PULSE_GROUP_NAMES'] 
            
        # hdf5 chunk size - as a tuple - columns are 1d
        self.hdf5ChunkSize = (DEFAULT_HDF5_CHUNK_SIZE,)
        if 'HDF5_CHUNK_SIZE' in userClass.lidarDriverOptions:
            self.hdf5ChunkSize = (userClass.lidarDriverOptions['HDF5_CHUNK_SIZE'],)                
                
        # attempt to open the file
        try:
            self.fileHandle = h5py.File(fname, h5py_mode)
        except (OSError, IOError) as err:
            # always seems to throw an OSError
            # found another one!
            raise generic.LiDARFormatNotUnderstood(str(err))
                
        # check that it is indeed the right version
        # and get attributes
        self.fileAttrs = self.fileHandle.attrs
        if mode == generic.READ or mode == generic.UPDATE:
            # not sure if this is ok - just check there are some header fields
            for expected in EXPECTED_HEADER_FIELDS:
                if expected not in self.fileHandle.attrs:
                    self.fileHandle = None
                    msg = '%s not found in header' % expected
                    raise generic.LiDARFormatNotUnderstood(msg)
        else:
            # add header values for current L1A01 format
            self.fileAttrs['l0_to_l1a_githash'] = b'19d6178607a7be37e67811a2d4f3fc4e145a335c'
            self.fileAttrs['l0_to_l1a_version'] = b'20190417.0.0'
        
        # get the ancillary data
        # TODO: Will the ancillary group ever need to change?
        self.ancillary = numpy.array([1.19880002e+09], dtype=[('master_time_epoch', '<f8')])
        if mode == generic.READ or mode == generic.UPDATE:
            if self.beam in self.fileHandle:
                self.ancillary = self.readAncillaryData(self.fileHandle, self.beam)
        
        # create new groups if necessary
        if mode == generic.CREATE or mode == generic.UPDATE:
            
            if self.beam not in self.fileHandle:
                data = self.fileHandle.create_group(self.beam)
                data.create_group('ancillary')
                
            for group_name in self.pulse_group_names:
                if group_name not in self.fileHandle[self.beam]:
                    data.create_group(group_name)                    

            # add the GENERATING_SOFTWARE tag
            self.fileAttrs['generating_software'] = generic.SOFTWARE_NAME
                
        self.range = None
        self.lastPulsesSpace = None
        self.lastTransSpace = None
        self.lastTrans_Idx = None
        self.lastTrans_IdxMask = None
        self.lastRecvSpace = None
        self.lastRecv_Idx = None
        self.lastRecv_IdxMask = None
    
    @staticmethod        
    def getDriverName():
        return 'GEDIL1A01'

    def close(self):
        
        if self.mode != generic.READ:
            self.setHeader(self.fileAttrs)
            self.setAncillary(self.ancillary)
        
        self.fileHandle = None
        self.range = None
        self.lastPulsesSpace = None
        self.lastTransSpace = None
        self.lastTrans_Idx = None
        self.lastTrans_IdxMask = None
        self.lastRecvSpace = None
        self.lastRecv_Idx = None
        self.lastRecv_IdxMask = None 

    def hasSpatialIndex(self):
        "GEDI L1A01 does not have a spatial index"
        return False

    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        self.range = copy.copy(pulseRange)
        nTotalPulses = self.getTotalNumberPulses()
        bMore = True
        if self.range.startPulse >= nTotalPulses:
            # no data to read
            self.range.startPulse = 0
            self.range.endPulse = 0
            bMore = False
            
        elif self.range.endPulse >= nTotalPulses:
            self.range.endPulse = nTotalPulses
            
        return bMore

    def readRange(self, colNames=None):
        """
        Internal method. Returns the requested column(s) as
        a structured array.
        Assumes colName is not None
        """
        # Must be a list of column names
        if isinstance(colNames, str):
            colNames = [colNames]
        
        dtypeList = []
        for name in colNames:            
            if name in self.fileHandle[self.beam]:
                s = self.fileHandle[self.beam][name].dtype.str
                if self.fileHandle[self.beam][name].ndim > 1:
                    t = self.fileHandle[self.beam][name].shape[0:-1]
                    dtypeList.append((str(name), s, t))
                else:
                    dtypeList.append((str(name), s))
            else:
                group_id = name[0:3]
                if group_id in GROUP_NAMES:
                    hdfName = name[4::]
                    if hdfName in self.fileHandle[self.beam][GROUP_NAMES[group_id]]:
                        s = self.fileHandle[self.beam][GROUP_NAMES[group_id]][hdfName].dtype.str
                        if self.fileHandle[self.beam][GROUP_NAMES[group_id]][hdfName].ndim > 1:
                            t = self.fileHandle[self.beam][GROUP_NAMES[group_id]][hdfName].shape[0:-1]
                            dtypeList.append((str(name), s, t))
                        else:
                            dtypeList.append((str(name), s))
                    else:
                        if name == CLASSIFICATION_NAME:
                            dtypeList.append((CLASSIFICATION_NAME, numpy.uint8))
                        else:
                            msg = 'column %s not found in file' % name
                            raise generic.LiDARArrayColumnError(msg)
                
        numRecords = self.range.endPulse - self.range.startPulse
        data = numpy.empty(numRecords, dtypeList)
        for name in colNames:
            if name in self.fileHandle[self.beam]:
                if self.fileHandle[self.beam][name].ndim > 1:
                    d = self.fileHandle[self.beam][name][...,self.range.startPulse:self.range.endPulse]
                    data[str(name)] = numpy.transpose(d)
                else:
                    data[str(name)] = self.fileHandle[self.beam][name][...,self.range.startPulse:self.range.endPulse]
            else:
                group_id = name[0:3]
                if group_id in GROUP_NAMES:
                    hdfName = name[4::]
                    if hdfName in self.fileHandle[self.beam][GROUP_NAMES[group_id]]:
                        if self.fileHandle[self.beam][GROUP_NAMES[group_id]][hdfName].ndim > 1:
                            d = self.fileHandle[self.beam][GROUP_NAMES[group_id]][hdfName][...,self.range.startPulse:self.range.endPulse]
                            data[str(name)] = numpy.transpose(d)
                        else:
                            data[str(name)] = self.fileHandle[self.beam][GROUP_NAMES[group_id]][hdfName][...,self.range.startPulse:self.range.endPulse]
                    else:
                        if name == CLASSIFICATION_NAME:
                            data[CLASSIFICATION_NAME].fill(0)
        
        return data

    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        # we only accept 'X', 'Y', 'Z' and do the translation 
        # from the self.pointFrom names
        dictn = {'X' : self.pointFrom[0], 'Y' : self.pointFrom[1], 'Z' : self.pointFrom[2], CLASSIFICATION_NAME : CLASSIFICATION_NAME}

        if colNames is None:
            colNames = ['X', 'Y', 'Z', CLASSIFICATION_NAME]

        if isinstance(colNames, str):
            # translate
            tranColName = dictn[colNames]
            # no need to translate on output as not a structured array
            data = self.readRange(tranColName)
        else:
            # a list. Do the translation
            tranColNames = [dictn[colName] for colName in colNames]

            # get the structured array
            data = self.readRange(tranColNames)

            # rename the columns to make it match requested names
            data.dtype.names = colNames

        return data
        
    def readPulsesForRange(self, colNames=None):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """        
        if colNames is None:
            
            colNames = []
            
            for name in self.fileHandle[self.beam]:
                if isinstance(self.fileHandle[self.beam][name], h5py.Dataset):
                    if name in MAIN_FIELDS:
                        colNames.append(str(name))
            
            for group in self.pulse_group_names:
                for hdfName in self.fileHandle[self.beam][group]:
                    if isinstance(self.fileHandle[self.beam][group][hdfName], h5py.Dataset):                   
                        if (hdfName in CLK_FIELDS or hdfName in INST_HK_FIELDS or hdfName in TX_PROCESSING_FIELDS or
                            hdfName in GEOLOCATION_FIELDS or hdfName in GEOPHYS_CORR_FIELDS):
                            name = "%s_%s" % (GROUP_IDS[group], hdfName)
                            colNames.append(str(name))
        
        return self.readRange(colNames)
        
    def readWaveformInfo(self):
        """
        A structured array containing information about the waveforms.
        """
        
        colNames = WAVEFORMINFO_FIELDS.keys()
        
        try:            
            waveformInfo = self.readRange(colNames)
        except generic.LiDARArrayColumnError:
            return None
                
        return waveformInfo   
       
    def readTransmitted(self):
        """
        Return the 3d masked integer array of transmitted for each of the
        current pulses.
        First axis is the waveform bin.
        Second axis is waveform number and last is pulse.
        """
        waveformInfo = self.readWaveformInfo()
        if waveformInfo is None:
            return None       
        
        if 'txwaveform' not in self.fileHandle[self.beam]:
            return None
        
        # read as 2d
        nOut = self.fileHandle[self.beam]['txwaveform'].shape[0]
        trans_space, trans_idx, trans_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        waveformInfo['tx_sample_start_index'], waveformInfo['tx_sample_count'], 
                                        nOut)
        
        start = int(waveformInfo['tx_sample_start_index'][0])
        finish = int(waveformInfo['tx_sample_start_index'][-1] + waveformInfo['tx_sample_count'][-1])
        trans = self.fileHandle[self.beam]['txwaveform'][start:finish]
        
        trans = trans[trans_idx]
        
        self.lastTransSpace = trans_space
        if self.mode == generic.UPDATE:
            self.lastTrans_Idx = trans_idx
            self.lastTrans_IdxMask = trans_idx_mask
        
        return numpy.ma.array(trans, mask=trans_idx_mask)
        
    def readReceived(self):
        """
        Return the 2d masked integer array of received for each of the
        current pulses.
        First axis is waveform number and last is pulse.
        """
        waveformInfo = self.readWaveformInfo()
        if waveformInfo is None:
            return None
        
        if 'rxwaveform' not in self.fileHandle[self.beam]:
            return None
            
        # read as 2d
        nOut = self.fileHandle[self.beam]['rxwaveform'].shape[0]
        recv_space, recv_idx, recv_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                      waveformInfo['rx_sample_start_index'], waveformInfo['rx_sample_count'], 
                                      nOut)
        
        start = int(waveformInfo['rx_sample_start_index'][0])
        finish = int(waveformInfo['rx_sample_start_index'][-1] + waveformInfo['rx_sample_count'][-1])
        recv = self.fileHandle[self.beam]['rxwaveform'][start:finish]
        
        recv = recv[recv_idx]
        
        self.lastRecvSpace = recv_space
        if self.mode == generic.UPDATE:
            self.lastRecv_Idx = recv_idx
            self.lastRecv_IdxMask = recv_idx_mask
        
        return numpy.ma.array(recv, mask=recv_idx_mask)
        
    def getTotalNumberPulses(self):
        """
        Return the total number of pulses
        """
        try:
            nPulses = self.fileHandle[self.beam]['shot_number'].shape[0]
        except (AttributeError, KeyError) as e:
            nPulses = 0

        return nPulses

    @staticmethod
    def readHeaderAsDict(fileHandle):
        """
        Internal method to gather info from file and build
        into a dictionary.
        """
        # return the stuff in the attrs
        header = {}
        for name in fileHandle.attrs:
            value = fileHandle.attrs[name][0]
            if sys.version_info[0] == 3 and isinstance(value, bytes):
                value = value.decode()
            header[str(name)] = value
            
        return header

    @staticmethod
    def readSummaryData(fileHandle, beam):
        """
        These are custom summary fields in the L1A01 files that do not have per pulse data
        Presently from the TX_PROCESSING group only
        """
        # create an empty structured array
        dtypeList = []
        for hdfName in TX_PROCESSING_SUMMARY_FIELDS:
            if hdfName in fileHandle[beam]['TX_PROCESSING']:
                s = fileHandle[beam]['TX_PROCESSING'][hdfName].dtype.str 
                t = fileHandle[beam]['TX_PROCESSING'][hdfName].shape
                dtypeList.append((str(hdfName), s, t))
        data = numpy.empty(1, dtype=dtypeList)
        
        # populate the array
        for hdfName in TX_PROCESSING_SUMMARY_FIELDS:
            if hdfName in fileHandle[beam]['TX_PROCESSING']:
                data[hdfName] = fileHandle[beam]['TX_PROCESSING'][hdfName][:]
        
        return data        

    @staticmethod
    def readAncillaryData(fileHandle, beam):
        """
        Read the ancillary data in the L1A01 files
        """
        # create an empty structured array
        dtypeList = []
        for hdfName in ANCILLARY_FIELDS:
            if hdfName in fileHandle[beam]['ancillary']:
                s = fileHandle[beam]['ancillary'][hdfName].dtype.str 
                t = fileHandle[beam]['ancillary'][hdfName].shape
                dtypeList.append((str(hdfName), s, t))
        data = numpy.empty(1, dtype=dtypeList)
        
        # populate the array
        for hdfName in ANCILLARY_FIELDS:
            if hdfName in fileHandle[beam]['ancillary']:
                data[hdfName] = fileHandle[beam]['ancillary'][hdfName][:]
        
        return data
        
    def prepareTransmittedForWriting(self, transmitted):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if transmitted.size == 0:
            return None, None, None

        if transmitted.ndim != 2:
            msg = 'transmitted data must be 2d'
            raise generic.LiDARInvalidData(msg)
        
        trans_start = None
        ntrans = None

        transmitted = numpy.ma.MaskedArray(transmitted, mask=transmitted.mask)
        
        if self.mode == generic.UPDATE:

            origShape = transmitted.shape

            # flatten it back to 1d so it can be written
            flatSize = self.lastTrans_Idx.max() + 1
            flatTrans = numpy.empty((flatSize,), dtype=transmitted.data.dtype)
            ntrans = numpy.empty((flatSize,), dtype=numpy.uint16)
            
            gridindexutils.flatten2dMaskedArray(flatTrans, transmitted, self.lastTrans_IdxMask, ntrans)   
            #flatten2dWaveformData(transmitted, self.lastTrans_IdxMask, ntrans, flatTrans)
            
            transmitted = flatTrans
                
        else:       
        
            # create arrays for flatten2dWaveformData
            ntrans = numpy.zeros(transmitted.shape[1], dtype=numpy.uint16)
            flattened = numpy.empty(transmitted.count(), dtype=transmitted.dtype)

            #flatten2dWaveformData(transmitted.data, transmitted.mask, ntrans, flattened)
            gridindexutils.flatten2dMaskedArray(flattened, transmitted.data, transmitted.mask, ntrans)
            
            currTransCount = 0
            if 'txwaveform' in self.fileHandle[self.beam]:
                transHandle = self.fileHandle[self.beam]['txwaveform']
                currTransCount = transHandle.shape[0]

            trans_start = numpy.cumsum(ntrans)
            trans_start = numpy.roll(trans_start, 1)
            if trans_start.size > 0:
                trans_start[0] = 0
            trans_start += currTransCount            
            
            transmitted = flattened
            
        return transmitted, trans_start, ntrans
    
    def prepareReceivedForWriting(self, received):
        """
        Called from writeData(). Massages what the user has passed into something
        we can write back to the file.
        """
        if received.size == 0:
            return None, None, None
        
        if received.ndim != 2:
            msg = 'received data must be 2d'
            raise generic.LiDARInvalidData(msg)
        
        recv_start = None
        nrecv = None
            
        received = numpy.ma.MaskedArray(received, mask=received.mask)

        if self.mode == generic.UPDATE:

            origShape = received.shape
            
            # flatten it back to 1d so it can be written
            flatSize = self.lastRecv_Idx.max() + 1
            flatRecv = numpy.empty((flatSize,), dtype=received.data.dtype)
            nrecv = numpy.empty((flatSize,), dtype=numpy.uint16)
            
            gridindexutils.flatten2dMaskedArray(flatRecv, received, self.lastRecv_IdxMask, nrecv)            
            #flatten2dWaveformData(received, self.lastRecv_IdxMask, nrecv, flatRecv)
            
            received = flatRecv
                
        else:
    
            # create arrays for flatten2dWaveformData
            nrecv = numpy.zeros(received.shape[1], dtype=numpy.uint16)
            flattened =  numpy.empty(received.count(), dtype=received.dtype)
            
            gridindexutils.flatten2dMaskedArray(flattened, received.data, received.mask, nrecv)
            #flatten2dWaveformData(received.data, received.mask, nrecv, flattened)
            
            currRecvCount = 0
            if 'rxwaveform' in self.fileHandle[self.beam]:
                recvHandle = self.fileHandle[self.beam]['rxwaveform']
                currRecvCount = recvHandle.shape[0]
            
            recv_start = numpy.cumsum(nrecv)
            recv_start = numpy.roll(recv_start, 1)
            if recv_start.size > 0:
                recv_start[0] = 0
            recv_start += currRecvCount            
                
            received = flattened

        return received, recv_start, nrecv        
        
    def createDataColumn(self, groupHandle, name, data):
        """
        Creates a new data column under groupHandle with the
        given name with standard HDF5 params.
        
        The type is the same as the numpy array data and data
        is written to the column

        sets the chunk size to self.hdf5ChunkSize which can be
        overridden in the driver options.
        """
        hdf5ChunkSize = list(data.shape)
        hdf5ChunkSize[0] = self.hdf5ChunkSize[0]
        hdf5ChunkSize = tuple(hdf5ChunkSize)
        dset = groupHandle.create_dataset(name, data.shape, 
                chunks=hdf5ChunkSize, dtype=data.dtype, shuffle=True, 
                compression="gzip", compression_opts=1, maxshape=(None,)*data.ndim)
        dset[...,:] = data
        
    def prepareDataForWriting(self, data, name):
        """
        Prepares data for writing to a field.            
        Returns the data to write.
        """       
        if name[0:3] in GROUP_NAMES:
            group_id = name[0:3]
            group = GROUP_NAMES[group_id]
            hdfName = name[4::]            
            if group_id == 'CLK':
                dataType = CLK_FIELDS[hdfName]
            elif group_id == 'IHK':
                dataType = INST_HK_FIELDS[hdfName]   
            elif group_id == 'TXP':
                dataType = TX_PROCESSING_FIELDS[hdfName]
            elif group_id == 'GEO':
                dataType = GEOLOCATION_FIELDS[hdfName] 
            elif group_id == 'GCO':
                dataType = GEOPHYS_CORR_FIELDS[hdfName]
                
        else:
            group = None
            hdfName = name
            dataType = MAIN_FIELDS[hdfName]
        
        # cast to datatype
        if numpy.issubdtype(dataType, numpy.integer):
            # check range
            info = numpy.iinfo(dataType)
            dataMin = data.min()
            if dataMin < info.min:
                msg = ('The data for field %s (%f) is less than ' +
                    'the minimum for the data type (%d)') % (name, dataMin, 
                    info.min)
                raise generic.LiDARScalingError(msg)

            dataMax = data.max()
            if dataMax > info.max:
                msg = ('The data for field %s (%f) is greater than ' +
                    'the maximum for the data type (%d)') % (name, dataMax,
                    info.max)
                raise generic.LiDARScalingError(msg)

            data = numpy.around(data).astype(dataType)
            
            if data.ndim > 1:
                data = numpy.transpose(data)
            
        return data, group, hdfName

    def writeStructuredArray(self, hdfHandle, structArray):
        """
        Writes a structured array as named datasets under hdfHandle
        Only use for file creation.
        """       
        for name in structArray.dtype.names:       
            if name not in WAVEFORMINFO_FIELDS:
                data, group, hdfname = self.prepareDataForWriting(
                            structArray[name], name)
                
                if group is None:
                    groupHandle = self.fileHandle[self.beam]
                else:
                    groupHandle = self.fileHandle[self.beam][group]
                
                if hdfname in groupHandle:
                    newSize = groupHandle[hdfname].shape[0] + len(structArray)
                    a = groupHandle[hdfname].ndim - 1
                    groupHandle[hdfname].resize(newSize, axis=a)
                    groupHandle[hdfname][...,oldSize:newSize+1] = data
                else:
                    self.createDataColumn(groupHandle, hdfname, data)

    def writeData(self, pulses=None, points=None, transmitted=None, received=None, waveformInfo=None):
        """
        Write all the updated data. Pass None for data that do not need to be updated.
        It is assumed that each parameter has been read by the reading functions
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return

        elif self.mode == generic.CREATE:
            
            if pulses is None:
                msg = 'Must provide pulses when writing new data'
                raise generic.LiDARInvalidData(msg)
                
            if pulses.ndim != 1:
                msg = 'pulses must be 1d as returned from getPulses'
                raise generic.LiDARInvalidData(msg)
            
            if waveformInfo is not None and waveformInfo.ndim != 1:
                msg = 'waveformInfo must be 1d as returned from getWaveformInfo'
                raise generic.LiDARInvalidData(msg)                        
            
            if transmitted is not None and transmitted.ndim != 2:
                msg = 'transmitted must be 2d as returned by readTransmitted'
                raise generic.LiDARInvalidData(msg)
            
            if received is not None and received.ndim != 2:
                msg = 'received must be 2d as returned by readReceived'
                raise generic.LiDARInvalidData(msg)
        
        if self.mode == generic.CREATE:

            if pulses is not None and len(pulses) > 0:
                self.writeStructuredArray(self.fileHandle[self.beam], pulses)
            
            if transmitted is not None and len(transmitted) > 0:
                transmitted, trans_start, ntrans = self.prepareTransmittedForWriting(transmitted)
                if 'txwaveform' in self.fileHandle[self.beam]:
                    tHandle = self.fileHandle[self.beam]['txwaveform']
                    oldSize = tHandle.shape[0]
                    newSize = oldSize + len(transmitted)
                    tHandle.resize((newSize,))
                    tHandle[oldSize:newSize+1] = transmitted
                else:
                    self.createDataColumn(self.fileHandle[self.beam], 
                                'txwaveform', transmitted)
                
                if 'tx_sample_start_index' in self.fileHandle[self.beam]:
                    tHandle = self.fileHandle[self.beam]['tx_sample_start_index']
                    oldSize = tHandle.shape[0]
                    newSize = oldSize + len(trans_start)
                    tHandle.resize((newSize,))
                    tHandle[oldSize:newSize+1] = trans_start
                else:
                    self.createDataColumn(self.fileHandle[self.beam], 
                                'tx_sample_start_index', trans_start)
                                
                if 'tx_sample_count' in self.fileHandle[self.beam]:
                    tHandle = self.fileHandle[self.beam]['tx_sample_count']
                    oldSize = tHandle.shape[0]
                    newSize = oldSize + len(ntrans)
                    tHandle.resize((newSize,))
                    tHandle[oldSize:newSize+1] = ntrans                    
                else:
                    self.createDataColumn(self.fileHandle[self.beam], 
                                'tx_sample_count', ntrans)           
            
            if received is not None and len(received) > 0:
                received, recv_start, nrecv = self.prepareReceivedForWriting(received)
                
                if 'rxwaveform' in self.fileHandle[self.beam]:
                    rHandle = self.fileHandle[self.beam]['rxwaveform']
                    oldSize = rHandle.shape[0]
                    newSize = oldSize + len(received)
                    rHandle.resize((newSize,))
                    rHandle[oldSize:newSize+1] = received
                else:
                    self.createDataColumn(self.fileHandle[self.beam], 
                                'rxwaveform', received)

                if 'rx_sample_start_index' in self.fileHandle[self.beam]:
                    rHandle = self.fileHandle[self.beam]['rx_sample_start_index']
                    oldSize = rHandle.shape[0]
                    newSize = oldSize + len(recv_start)
                    rHandle.resize((newSize,))
                    rHandle[oldSize:newSize+1] = recv_start                    
                else:
                    self.createDataColumn(self.fileHandle[self.beam], 
                                'rx_sample_start_index', recv_start)

                if 'rx_sample_count' in self.fileHandle[self.beam]:
                    rHandle = self.fileHandle[self.beam]['rx_sample_count']
                    oldSize = rHandle.shape[0]
                    newSize = oldSize + len(nrecv)
                    rHandle.resize((newSize,))
                    rHandle[oldSize:newSize+1] = nrecv                    
                else:
                    self.createDataColumn(self.fileHandle[self.beam], 
                                'rx_sample_count', nrecv)
                                
        else:
            
            if pulses is not None:
                for name in pulses.dtype.names:
                    if name not in WAVEFORMINFO_FIELDS:
                        data, group, hdfname = self.prepareDataForWriting(pulses[name], name)                   
                        if data.size > 0:
                            if group is None:
                                groupHandle = self.fileHandle[self.beam]
                            else:
                                groupHandle = self.fileHandle[self.beam][group]
                            if hdfname in groupHandle:
                                # get: Array must be C-contiguous without the copy
                                self.lastPulsesSpace.write(groupHandle[hdfname], data.copy())
                            else:
                                self.createDataColumn(groupHandle, name, data)
                            
            if transmitted is not None:
                transmitted, trans_start, ntrans = self.prepareTransmittedForWriting(transmitted)
                self.lastTransSpace.write(self.fileHandle[self.beam]['txwaveform'], transmitted)
                data, group, hdfname = self.prepareDataForWriting(trans_start, 'tx_sample_start_index')
                if data.size > 0:
                    if hdfname in self.fileHandle[self.beam]:
                        # get: Array must be C-contiguous without the copy
                        self.lastPulsesSpace.write(self.fileHandle[self.beam][hdfname], data.copy())
                    else:
                        self.createDataColumn(self.fileHandle[self.beam], hdfname, data)
                data, group, hdfname = self.prepareDataForWriting(ntrans, 'tx_sample_count')
                if data.size > 0:
                    if hdfname in self.fileHandle[self.beam]:
                        # get: Array must be C-contiguous without the copy
                        self.lastPulsesSpace.write(self.fileHandle[self.beam][hdfname], data.copy())
                    else:
                        self.createDataColumn(self.fileHandle[self.beam], hdfname, data)
                        
            if received is not None:
                received, recv_start, nrecv = self.prepareReceivedForWriting(received)
                self.lastRecvSpace.write(self.fileHandle[self.beam]['rxwaveform'], received)
                data, group, hdfname = self.prepareDataForWriting(recv_start, 'rx_sample_start_index')
                if data.size > 0:
                    if hdfname in self.fileHandle[self.beam]:
                        # get: Array must be C-contiguous without the copy
                        self.lastPulsesSpace.write(self.fileHandle[self.beam][hdfname], data.copy())
                    else:
                        self.createDataColumn(self.fileHandle[self.beam], hdfname, data)
                data, group, hdfname = self.prepareDataForWriting(nrecv, 'rx_sample_count')
                if data.size > 0:
                    if hdfname in self.fileHandle[self.beam]:
                        # get: Array must be C-contiguous without the copy
                        self.lastPulsesSpace.write(self.fileHandle[self.beam][hdfname], data.copy())
                    else:
                        self.createDataColumn(self.fileHandle[self.beam], hdfname, data)
              
    def getHeader(self):
        """
        Get the header as a dictionary
        """
        return self.readHeaderAsDict(self.fileHandle)

    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]

    def setHeader(self, newHeaderDict):
        """
        Update our cached dictionary
        """
        if self.mode == generic.READ:
            msg = 'Can only set header values on update or create'
            raise generic.LiDARInvalidSetting(msg)
        for key in newHeaderDict.keys():
            self.fileHandle.attrs[key] = newHeaderDict[key]
        
    def setHeaderValue(self, name, value):
        """
        Just update one value in the header
        """
        if self.mode == generic.READ:
            msg = 'Can only set header values on update or create'
            raise generic.LiDARInvalidSetting(msg)
        self.fileHandle.attrs[name] = value
        
    def setAncillary(self, newAncillary):
        """
        Update our cached structured array
        """
        if self.mode == generic.READ:
            msg = 'Can only set ancillary data on update or create'
            raise generic.LiDARInvalidSetting(msg)
        
        for hdfName in ANCILLARY_FIELDS:
            if hdfName in newAncillary.dtype.names:
                if hdfName in self.fileHandle[self.beam]['ancillary']:
                    self.fileHandle[self.beam]['ancillary'][hdfName][...,:] = newAncillary[hdfName]
                else:
                    self.createDataColumn(self.fileHandle[self.beam]['ancillary'], hdfName, newAncillary[hdfName])
        
    
class GEDIL1A01FileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a GEDI file
    and makes it available as fields.
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)
        
        # attempt to open the file
        try:
            fileHandle = h5py.File(fname, 'r')
        except (OSError, IOError) as err:
            # always seems to throw an OSError
            # found another one!
            raise generic.LiDARFormatNotUnderstood(str(err))

        # not sure if this is ok - just check there are some header fields
        for expected in EXPECTED_HEADER_FIELDS:
            if expected not in fileHandle.attrs:
                msg = '%s not found in header' % expected
                raise generic.LiDARFormatNotUnderstood(msg)
        
        # read the ancillary and summary data
        self.header = GEDIL1A01File.readHeaderAsDict(fileHandle)        
        self.ancillary = {}
        self.summary = {}
        for beam in fileHandle.keys():       
            self.ancillary[beam] = GEDIL1A01File.readAncillaryData(fileHandle, beam)
            self.summary[beam] = GEDIL1A01File.readSummaryData(fileHandle, beam)
        
            
    @staticmethod        
    def getDriverName():
        return 'GEDIL1A01'