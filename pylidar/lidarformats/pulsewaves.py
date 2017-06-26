"""
Driver for PulseWaves.

Read Driver Options
-------------------

These are contained in the READSUPPORTEDOPTIONS module level variable.

+-----------------------+---------------------------------------------+
| Name                  | Use                                         |
+=======================+=============================================+
| POINT_FROM            | an integer. Set to one of the POINT_FROM_*  |
|                       | module level constants. Determines which    |
|                       | file the coordinates for the point is       |
|                       | created from. Defaults to POINT_FROM_ANCHOR |
+-----------------------+---------------------------------------------+
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

import os
import copy
import numpy
import datetime

from . import generic
from . import gridindexutils
# Fail slightly less drastically when running from ReadTheDocs
if os.getenv('READTHEDOCS', default='False') != 'True':
    from . import _pulsewaves
    # bring constants over
    POINT_FROM_ANCHOR = _pulsewaves.POINT_FROM_ANCHOR
    POINT_FROM_TARGET = _pulsewaves.POINT_FROM_TARGET
    "How the points are set"
else:
    POINT_FROM_ANCHOR = None
    POINT_FROM_TARGET = None
    "How the points are set"

READSUPPORTEDOPTIONS = ('POINT_FROM',)
"Supported read options"

today = datetime.date.today()
DEFAULT_HEADER = {"GENERATING_SOFTWARE" : generic.SOFTWARE_NAME, 
"FILE_CREATION_DAY" : today.toordinal() - datetime.date(today.year, 1, 1).toordinal(), 
"FILE_CREATION_YEAR" : today.year}
"for new files"

def isPulseWavesFile(fname):
    """
    Helper function that looks at the start of the file
    to determine if it is a pulsewaves file or not
    """
    # See if it starts with 'PulseWavesPulse'. The library won't open the .wvs file
    # - only the .pls
    fh = open(fname, 'rb')
    data = fh.read(15)
    fh.close()
        
    if data != b'PulseWavesPulse':
        return False
    else:
        return True

class PulseWavesFile(generic.LiDARFile):
    """
    Reader/Writer for PulseWaves files
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)    

        if mode != generic.READ and mode != generic.CREATE:
            msg = 'PulseWaves driver is read or create only'
            raise generic.LiDARInvalidSetting(msg)

        for key in userClass.lidarDriverOptions:
            if key not in READSUPPORTEDOPTIONS:
                msg = '%s not a supported pulsewaves option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)

        if mode == generic.READ and not isPulseWavesFile(fname):
            msg = 'not a PulseWaves file'
            raise generic.LiDARFileException(msg)

        point_from = POINT_FROM_ANCHOR
        if 'POINT_FROM' in userClass.lidarDriverOptions:
            point_from = userClass.lidarDriverOptions['POINT_FROM']

        if mode == generic.READ:
            # read
            try:
                self.pulsewavesFile = _pulsewaves.FileRead(fname, point_from)
            except _pulsewaves.error:
                msg = 'error opening pulsewaves file'
                raise generic.LiDARFormatNotUnderstood(msg)

        else:
            # create
            try:
                self.pulsewavesFile = _pulsewaves.FileWrite(fname)
            except _pulsewaves.error as e:
                msg = 'cannot create pulsewaves file' + str(e)
                raise generic.LiDARFileException(msg)

        if mode == generic.READ:
            self.header = None
        else:
            self.header = DEFAULT_HEADER
            
        self.range = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None
        self.lastTransmitted = None
        self.firstBlockWritten = False # can't write header values when this is True

    @staticmethod        
    def getDriverName():
        return 'PulseWaves'

    def close(self):
        self.pulsewavesFile = None
        self.range = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveformInfo = None
        self.lastReceived = None
        self.lastTransmitted = None

    def readPointsByPulse(self, colNames=None):
        """
        Return a 2d masked structured array of point that matches
        the pulses.
        """
        # just read the points and add a dimensions
        # since there is one point per pulse
        points = self.readPointsForRange(colNames)
        points = numpy.expand_dims(points, 0)
        mask = numpy.zeros_like(points, dtype=numpy.bool)

        return numpy.ma.array(points, mask=mask)

    def hasSpatialIndex(self):
        "PulseWaves does not have a spatial index"
        return False

    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        self.range = copy.copy(pulseRange)
        # return True if we can still read data
        # we just assume we can until we find out
        # after a read that we can't
        return not self.pulsewavesFile.finished
    
    def readData(self, extent=None):
        """
        Internal method. Just reads into the self.last* fields
        """
        pulses, points, info, recv, trans = self.pulsewavesFile.readData(
                    self.range.startPulse, self.range.endPulse)
        self.lastRange = copy.copy(self.range)
        self.lastPoints = points
        self.lastPulses = pulses
        self.lastWaveformInfo = info
        self.lastReceived = recv
        self.lastTransmitted = trans

    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        self.readData()                            
        return self.subsetColumns(self.lastPoints, colNames)
        
    def readPulsesForRange(self, colNames=None):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        self.readData()                            
        return self.subsetColumns(self.lastPulses, colNames)
        
    def readWaveformInfo(self):
        """
        2d structured masked array containing information
        about the waveforms.
        """
        self.readData()                            
        # workaround - seems a structured array returned from
        # C doesn't work with masked arrays. The dtype looks different.
        # TODO: check this with a later numpy
        colNames = self.lastWaveformInfo.dtype.names
        info = self.subsetColumns(self.lastWaveformInfo, colNames)
        if info is not None:
            # TODO: cache?
            idx = self.lastPulses['WFM_START_IDX']
            cnt = self.lastPulses['NUMBER_OF_WAVEFORM_SAMPLES']

            # ok format the waveform info into a 2d (by pulse) structure using
            # the start and count fields (that connect with the pulse) 
            wave_idx, wave_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        idx, cnt)
            info = info[wave_idx]
            info = numpy.ma.array(info, mask=wave_idx_mask)

        return info

    def readTransmitted(self):
        # need info for the indexing into the waveforms
        info = self.readWaveformInfo()

        # TODO: cache?

        # now the waveforms. Use the just created 2d array of waveform info's to
        # create the 3d one. 
        idx = info['TRANSMITTED_START_IDX']
        cnt = info['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS']
            
        trans_idx, trans_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        idx, cnt)
        trans = self.lastTransmitted[trans_idx]
        trans = numpy.ma.array(trans, mask=trans_idx_mask)

        return trans
        
    def readReceived(self):
        # need info for the indexing into the waveforms
        info = self.readWaveformInfo()

        # TODO: cache?

        # now the waveforms. Use the just created 2d array of waveform info's to
        # create the 3d one. 
        idx = info['RECEIVED_START_IDX']
        cnt = info['NUMBER_OF_WAVEFORM_RECEIVED_BINS']
            
        recv_idx, recv_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        idx, cnt)
        recv = self.lastReceived[recv_idx]
        recv = numpy.ma.array(recv, mask=recv_idx_mask)

        return recv
        
    def getTotalNumberPulses(self):
        """
        Return the total number of pulses
        """
        return self.pulsewavesFile.numPulses

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        Write all the updated data. Pass None for data that do not need to be up
        It is assumed that each parameter has been read by the reading functions
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return

        if waveformInfo is not None:
            # must convert received to uint16. In theory liblas can handle
            # uint8 also, but for simplicity have made it uint16
            for waveform in range(waveformInfo.shape[0]):
                if received is not None:
                    gain = waveformInfo[waveform]['RECEIVE_WAVE_GAIN']
                    offset = waveformInfo[waveform]['RECEIVE_WAVE_OFFSET']
                    received[:,waveform] = (received[:,waveform] - gain) / offset

                if transmitted is not None:
                    gain = waveformInfo[waveform]['TRANS_WAVE_GAIN']
                    offset = waveformInfo[waveform]['TRANS_WAVE_OFFSET']
                    transmitted[:,waveform] = (transmitted[:,waveform] - gain) / offset
            
            if received is not None:
                received = received.astype(numpy.uint16)
            if transmitted is not None:
                transmitted = transmitted.astype(numpy.uint16)

        #print(pulses.shape, points.shape, received.shape, transmitted, waveformInfo.shape)
        #print(pulses.shape, points.shape, received.shape, transmitted.shape, waveformInfo.shape)
        # TODO: flatten if necessary
        self.pulsewavesFile.writeData(self.header, pulses, points, waveformInfo,
                                received, transmitted)
        self.firstBlockWritten = True

    def getHeader(self):
        """
        Get header as a dictionary
        """
        if self.header is None:
            self.header = self.pulsewavesFile.readHeader()
            
        return self.header

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
            msg = 'Can only set header values on create'
            raise generic.LiDARInvalidSetting(msg)
            
        if self.firstBlockWritten:
            msg = 'Header can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)
            
        for key in newHeaderDict.keys():
            self.header[key] = newHeaderDict[key]

    def setHeaderValue(self, name, value):
        """
        Just update one value in the header
        """
        if self.mode == generic.READ:
            msg = 'Can only set header values on create'
            raise generic.LiDARInvalidSetting(msg)

        if self.firstBlockWritten:
            msg = 'Header can only be updated before first block written'
            raise generic.LiDARFunctionUnsupported(msg)

        self.header[name] = value
    
class PulseWavesFileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a PulseWaves file
    and makes it available as fields.
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)

        if not isPulseWavesFile(fname):
            msg = 'not a pulsewaves file'
            raise generic.LiDARFormatNotUnderstood(msg)
        
        # open the file object
        try:
            pulsewavesFile = _pulsewaves.FileRead(fname)
        except _pulsewaves.error:
            msg = 'error opening pulsewaves file'
            raise generic.LiDARFormatNotUnderstood(msg)

        # get header
        self.header = pulsewavesFile.readHeader()

    @staticmethod        
    def getDriverName():
        return 'PulseWaves'
