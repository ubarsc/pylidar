"""
Driver for GEDI L1A01 HDF5 files. Read only.

Read Driver Options
-------------------

These are contained in the READSUPPORTEDOPTIONS module level variable.

+-----------------------+--------------------------------------------+
| Name                  | Use                                        |
+=======================+============================================+
| POINT_FROM            | A 3 element tuple defining which fields to |
|                       | create a fake point from (x,y,z). Default  |
|                       | is ('LON0', 'LAT0', 'Z0')                  |
+-----------------------+--------------------------------------------+
| BEAM                  | A string defining the beam id to use. One  |
|                       | of ['BEAM0000', 'BEAM0001', 'BEAM0010',    |
|                       | 'BEAM0011', 'BEAM0101', 'BEAM0110',        |
|                       | 'BEAM1000', 'BEAM1011']                    |
+-----------------------+--------------------------------------------+
"""

# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
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

from . import generic

READSUPPORTEDOPTIONS = ('POINT_FROM', 'BEAM')
"Supported read options"

DEFAULT_POINT_FROM = ('longitude_lastbin', 'latitude_lastbin', 'elevation_lastbin')
DEFAULT_BEAM = 'BEAM0101'
EXPECTED_HEADER_FIELDS = ['l0_to_l1a_githash', 'l0_to_l1a_version']
CLASSIFICATION_NAME = 'CLASSIFICATION'
"GEDI L1A01 files don't have a CLASSIFICATION column so we have to create a blank one for SPDV4"
SURFACE_TYPE_NAMES = ['land','ocean','sea_ice','land_ice','inland_water']

class GEDIL1A01File(generic.LiDARFile):
    """
    Reader for GEDI L1A01 files
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)    

        if mode != generic.READ:
            msg = 'GEDI L1A01 driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        for key in userClass.lidarDriverOptions:
            if key not in READSUPPORTEDOPTIONS:
                msg = '%s not a supported GEDI L1A01 option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)

        # attempt to open the file
        try:
            self.fileHandle = h5py.File(fname, 'r')
        except (OSError, IOError) as err:
            # always seems to throw an OSError
            # found another one!
            raise generic.LiDARFormatNotUnderstood(str(err))

        # not sure if this is ok - just check there are some header fields
        for expected in EXPECTED_HEADER_FIELDS:
            if expected not in self.fileHandle.attrs:
                self.fileHandle = None
                msg = '%s not found in header' % expected
                raise generic.LiDARFormatNotUnderstood(msg)

        # driver options
        self.pointFrom = DEFAULT_POINT_FROM
        if 'POINT_FROM' in userClass.lidarDriverOptions:
            self.pointFrom = userClass.lidarDriverOptions['POINT_FROM']
        self.beam = DEFAULT_BEAM
        if 'BEAM' in userClass.lidarDriverOptions:
            self.beam = userClass.lidarDriverOptions['BEAM']
        
        self.range = None

    @staticmethod        
    def getDriverName():
        return 'GEDIL1A01'

    def close(self):
        self.fileHandle = None
        self.range = None

    def readPointsByPulse(self, colNames=None):
        """
        Return a 2d masked structured array of point that matches
        the pulses.
        """
        # just read the points and add a dimensions
        # since there is one point per pulse
        points = self.readPointsForRange(colNames)
        points = numpy.expand_dims(points, 0)

        # make mask (can't just supply False as numpy gives an error)
        mask = numpy.zeros_like(points, dtype=numpy.bool)

        return numpy.ma.array(points, mask=mask)

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
        a structured array. Since both points and pulses come
        from the same place this function is called to read both.

        Assumes colName is not None
        """
        if isinstance(colNames, str):
            if colNames == CLASSIFICATION_NAME and colNames not in self.fileHandle[self.beam]['geolocation'] and colNames not in self.fileHandle[self.beam]:
                # hack so we can fake a CLASSIFICATION column
                numRecords = self.range.endPulse - self.range.startPulse
                return numpy.zeros(numRecords, dtype=numpy.uint8)

            if colNames in self.fileHandle[self.beam]:
                return self.fileHandle[self.beam][colNames][self.range.startPulse:self.range.endPulse]
            else:
                return self.fileHandle[self.beam]['geolocation'][colNames][self.range.startPulse:self.range.endPulse]
        else:
            # a list etc. Have to build structured array first
            dtypeList = []
            for name in colNames:
                if name == CLASSIFICATION_NAME and name not in self.fileHandle[self.beam]['geolocation'] and name not in self.fileHandle[self.beam]:
                    dtypeList.append((CLASSIFICATION_NAME, numpy.uint8))
                else:
                    if name in self.fileHandle[self.beam]:
                        s = self.fileHandle[self.beam][name].dtype.str
                        dtypeList.append((str(name), s))
                    elif name in self.fileHandle[self.beam]['geolocation']:
                        s = self.fileHandle[self.beam]['geolocation'][name].dtype.str
                        dtypeList.append((str(name), s))
                    else:
                        msg = 'column %s not found in file' % name
                        raise generic.LiDARArrayColumnError(msg)

            numRecords = self.range.endPulse - self.range.startPulse
            data = numpy.empty(numRecords, dtypeList)
            for name in colNames:
                if name == CLASSIFICATION_NAME and name not in self.fileHandle[self.beam]['geolocation'] and name not in self.fileHandle[self.beam]:
                    data[CLASSIFICATION_NAME].fill(0)
                else:
                    if name in SURFACE_TYPE_NAMES:
                        idx = SURFACE_TYPE_NAMES.index(name)
                        data[str(name)] = self.fileHandle[self.beam]['geolocation']['surface_type'][idx,self.range.startPulse:self.range.endPulse]
                    elif name in self.fileHandle[self.beam]:
                        data[str(name)] = self.fileHandle[self.beam][name][self.range.startPulse:self.range.endPulse]
                    else:
                        data[str(name)] = self.fileHandle[self.beam]['geolocation'][name][self.range.startPulse:self.range.endPulse]

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
            for name in self.fileHandle[self.beam]['geolocation'].keys():
                # add all the ones that are 1d array
                try:
                    # some may be sub-datasets etc
                    shape = self.fileHandle[self.beam]['geolocation'][name].shape
                except AttributeError as e:
                    continue

                if len(shape) == 1:
                    colNames.append(str(name))
                else:
                    if name == 'surface_type':
                        for surface_type_name in SURFACE_TYPE_NAMES:
                            colNames.append(str(surface_type_name))
            for name in self.fileHandle[self.beam].keys():
                # add all the ones that are 1d array
                try:
                    # some may be sub-datasets etc
                    shape = self.fileHandle[self.beam][name].shape
                except AttributeError as e:
                    continue

                if len(shape) == 1:
                    colNames.append(str(name))
                    
        return self.readRange(colNames)
        
    def readWaveformInfo(self):
        """
        2d structured masked array containing information
        about the waveforms.
        """
        # This is quite easy. Data is stored as a 2d array so just get the 
        # pulses we need. All data populated so mask is all False.
        if 'txwaveform' not in self.fileHandle[self.beam] or 'rxwaveform'  not in self.fileHandle[self.beam]:
            # TODO: check we always have both.
            return None

        numWf = self.fileHandle[self.beam]['shot_number'].shape[0]
        nPulses = self.range.endPulse - self.range.startPulse

        # create an empty structured array
        data = numpy.empty(nPulses, dtype=[('NUMBER_OF_WAVEFORM_RECEIVED_BINS', 'U16'),
                    ('RECEIVED_START_IDX', 'U64'), 
                    ('NUMBER_OF_WAVEFORM_TRANSMITTED_BINS', 'U16'), 
                    ('TRANSMITTED_START_IDX', 'U64'),
                    ('RECEIVE_WAVE_OFFSET', 'float32'), ('RECEIVE_WAVE_GAIN', 'float32'),
                    ('TRANS_WAVE_OFFSET', 'float32'), ('TRANS_WAVE_GAIN', 'float32')])
        
        data['NUMBER_OF_WAVEFORM_RECEIVED_BINS'] = self.fileHandle[self.beam]['rx_sample_count'][self.range.startPulse:self.range.endPulse]
        data['RECEIVED_START_IDX'] = self.fileHandle[self.beam]['rx_sample_start_index'][self.range.startPulse:self.range.endPulse]
        data['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS'] = self.fileHandle[self.beam]['tx_sample_count'][self.range.startPulse:self.range.endPulse]
        data['TRANSMITTED_START_IDX'] = self.fileHandle[self.beam]['tx_sample_start_index'][self.range.startPulse:self.range.endPulse]
        # need for SPDV4
        data['RECEIVE_WAVE_OFFSET'] = 0
        data['RECEIVE_WAVE_GAIN'] = 1
        data['TRANS_WAVE_OFFSET'] = 0
        data['TRANS_WAVE_GAIN'] = 1
        
        # make 2d
        wave_idx, wave_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                      data['RECEIVED_START_IDX'], data['NUMBER_OF_WAVEFORM_RECEIVED_BINS'])
        data = data[wave_idx]
        data = numpy.ma.array(data, mask=wave_idx_mask)

        return data      
        
    def readTransmitted(self):
        """
        Return the 3d masked integer array of transmitted for each of the
        current pulses.
        First axis is the waveform bin.
        Second axis is waveform number and last is pulse.
        """
        if 'txwaveform' not in self.fileHandle[self.beam]:
            return None
        
        # need info for the indexing into the waveforms
        info = self.readWaveformInfo()
        
        # read as 2d
        trans = self.fileHandle[self.beam]['txwaveform'][self.range.startPulse:self.range.endPulse]       
        trans_idx, trans_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        info['TRANSMITTED_START_IDX'], info['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS'])
        trans = trans[trans_idx]

        return numpy.ma.array(trans, mask=trans_idx_mask)
        
    def readReceived(self):
        """
        Return the 3d masked integer array of received for each of the
        current pulses.
        First axis is the waveform bin.
        Second axis is waveform number and last is pulse.
        """
        if 'rxwaveform' not in self.fileHandle[self.beam]:
            return None

        # need info for the indexing into the waveforms
        info = self.readWaveformInfo()
            
        # read as 2d
        recv = self.fileHandle[self.beam]['rxwaveform'][self.range.startPulse:self.range.endPulse]
        recv_idx, recv_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                      info['RECEIVED_START_IDX'], info['NUMBER_OF_WAVEFORM_RECEIVED_BINS'])
        recv = recv[recv_idx]

        return numpy.ma.array(recv, mask=recv_idx_mask)
        
    def getTotalNumberPulses(self):
        """
        Return the total number of pulses
        """
        try:
            nPulses = self.fileHandle[self.beam]['shot_number'].shape[0]
        except AttributeError as e:
            nPulses = 0

        return nPulses

    @staticmethod
    def readHeaderAsDict(fileHandle):
        """
        Internal method to gather info from file and build
        into a dictionary.
        """
        # return the stuff in the attrs and the ancillary_data
        header = {}
        for name in fileHandle.attrs:
            value = fileHandle.attrs[name][0]
            if sys.version_info[0] == 3 and isinstance(value, bytes):
                value = value.decode()
            header[str(name)] = value

        return header

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        Write all the updated data. Pass None for data that do not need to be up
        It is assumed that each parameter has been read by the reading functions
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return

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

        self.header = GEDIL1A01File.readHeaderAsDict(fileHandle)
            
    @staticmethod        
    def getDriverName():
        return 'GEDI L1A01'
