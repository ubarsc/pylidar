
"""
Driver for riegl rxp files. Note that the requires access to Riegl's 
'rivlib' and 'riwavelib' closed source libraries that can be obtained from Riegl.

To build Riegl support the $RIVLIB_ROOT and $RIWAVELIB_ROOT environment variables
must be set before running setup.py.

These variables must point into the directories that rivlib and riwavelib
created when they were unzipped.

At runtime, $RIWAVELIB_ROOT/lib must be added to your LD_LIBRARY_PATH (Unix)
or PATH (Windows) environment variable so the linked library can be found.

If you wish the waveforms to be processed then they must be extracted from 
the .rxp file into a .wfm file using the 'rxp2wfm' utility (supplied with 
the riwavelib). Best results are when the -i option (to build an index) is 
passed. Ensure that the output .wfm file has the same path as the input .rxp
file but different extension so that pylidar can find it. Here is an example
of running rxp2wfm:

$ rxp2wfm -i --uri data.rxp --out data.wfm

Driver Options
--------------

These are contained in the SUPPORTEDOPTIONS module level variable.

+-----------------------+---------------------------------------+
| Name                  | Use                                   |
+=======================+=======================================+
| ROTATION_MATRIX       | a 4x4 float32 array containing the    |
|                       | rotation to be applied to the data.   |
|                       | Can be obtained from RieglFileInfo.   |
+-----------------------+---------------------------------------+
| MAGNETIC_DECLINATION  | number of degrees                     |
+-----------------------+---------------------------------------+

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

from . import generic
# Fail slightly less drastically when running from ReadTheDocs
if os.getenv('READTHEDOCS', default='False') != 'True':
    from . import _riegl
    SUPPORTEDOPTIONS = _riegl.getSupportedOptions()
else:
    SUPPORTEDOPTIONS = {}

from . import gridindexutils

def isRieglFile(fname):
    """
    Helper function that looks at the start of the file
    to determine if it is a riegl file or not
    """
    # The riegl supplied functions only return an error when 
    # we actually start reading but PyLidar needs to know straight
    # away. Workaround is to read the start of the file and see
    # if it contains 'Riegl'. Normally at byte position 9 but we check
    # any of the first 32 bytes.
    fh = open(fname, 'rb')
    data = fh.read(32)
    fh.close()
        
    if data.find(b'Riegl') == -1:
        return False
    else:
        return True

class RieglFile(generic.LiDARFile):
    """
    Driver for reading Riegl rxp files. Uses rivlib
    via the _riegl module.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)
        
        if mode != generic.READ:
            msg = 'Riegl driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        if not isRieglFile(fname):
            msg = 'not a riegl file'
            raise generic.LiDARFileException(msg)
        
        # test if there is a .wfm file with the same base name
        self.haveWave = True
        root, ext = os.path.splitext(fname)
        waveName = root + '.wfm'
        if not os.path.exists(waveName):
            waveName = None
            self.haveWave = False
            msg = '.wfm file not found. Use "rxp2wfm -i" to extract first.'
            controls.messageHandler(msg, generic.MESSAGE_WARNING)
            
        # check if the options are all valid. Good to check in case of typo.
        # hard to do this in C
        for key in userClass.lidarDriverOptions:
            if key not in SUPPORTEDOPTIONS:
                msg = '%s not a supported Riegl option' % repr(key)
                raise generic.LiDARInvalidSetting(msg)
        
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveRange = None
        self.lastWaveInfo = None
        self.lastReceived = None
        self.header = None # only populate when asked for - slow operation
        self.scanFile = _riegl.ScanFile(fname, waveName, 
                        userClass.lidarDriverOptions)
                          
    @staticmethod        
    def getDriverName():
        return 'riegl'

    @staticmethod
    def getTranslationDict(arrayType):
        """
        Translation dictionary between formats
        """
        dict = {}
        if arrayType == generic.ARRAY_TYPE_POINTS:
            dict[generic.FIELD_POINTS_RETURN_NUMBER] = 'RETURN_NUMBER'
        return dict

    @staticmethod
    def getHeaderTranslationDict():
        """
        Nothing yet - empty dict
        """
        return {}
        
    def close(self):
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.lastWaveRange = None
        self.lastWaveInfo = None
        self.lastReceived = None
        self.scanFile = None
        
    def readPointsByPulse(self, colNames=None):
        """
        Read a 2d structured masked array containing the points
        for each pulse.
        """
        pulses = self.readPulsesForRange()
        points = self.readPointsForRange()
        if points.size == 0:
            return None
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']

        point_idx, point_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(        
                startIdxs, nReturns)
                
        pointsByPulse = points[point_idx]
        
        if colNames is None:
            # workaround - seems a structured array returned from
            # C doesn't work with masked arrays. The dtype looks different.
            # TODO: check this with a later numpy
            colNames = pointsByPulse.dtype.names
            
        pointsByPulse = self.subsetColumns(pointsByPulse, colNames)
        points = numpy.ma.array(pointsByPulse, mask=point_idx_mask)
        
        return points
        
    def readWaveforms(self):
        """
        Internal method. Returns both the waveform info
        and the received formatted in the appropriate way.
        """
        if not self.haveWave:
            return None, None
        
        if self.lastWaveRange is None or self.range != self.lastWaveRange:
            
            info, received, st, cnt = self.scanFile.readWaveforms(self.range.startPulse, 
                                    self.range.endPulse)
            self.lastWaveRange = self.range        
            
            # ok format the waveform info into a 2d (by pulse) structure using
            # the start and count fields (that connect with the pulse) 
            # returned from scanFile.readWaveforms
            wave_idx, wave_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        st, cnt)
            
            self.lastWaveInfo = info[wave_idx]
            # workaround - seems a structured array returned from
            # C doesn't work with masked arrays. The dtype looks different.
            # TODO: check this with a later numpy
            self.lastWaveInfo = self.subsetColumns(self.lastWaveInfo, 
                                    self.lastWaveInfo.dtype.names)
            
            self.lastWaveInfo = numpy.ma.array(self.lastWaveInfo, mask=wave_idx_mask)
            
            # now the waveforms. Use the just created 2d array of waveform info's to
            # create the 3d one. TODO: sort by channel?
            idx = self.lastWaveInfo['RECEIVED_START_IDX']
            cnt = self.lastWaveInfo['NUMBER_OF_WAVEFORM_RECEIVED_BINS']
            
            recv_idx, recv_idx_mask = gridindexutils.convertSPDIdxToReadIdxAndMaskInfo(
                                        idx, cnt)
            self.lastReceived = received[recv_idx]
            self.lastReceived = numpy.ma.array(self.lastReceived, mask=recv_idx_mask)
            
        return self.lastWaveInfo, self.lastReceived
        
    def readWaveformInfo(self):
        """
        3d structured masked array containing information
        about the waveforms.
        """
        waveInfo, recv = self.readWaveforms()

        return waveInfo
        
    def readTransmitted(self):
        """
        Riegl (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        """
        Read the received waveform for all pulses
        returns a 3d masked array
        First axis is the waveform bin.
        Second axis is waveform number and last is pulse.
        """
        waveInfo, recv = self.readWaveforms()
        
        return recv

    def hasSpatialIndex(self):
        """
        Riegl files aren't spatially indexed
        """
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
        return not self.scanFile.finished
    
    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        if self.lastRange is None or self.range != self.lastRange:
            pulses, points = self.scanFile.readData(self.range.startPulse, 
                            self.range.endPulse)
            self.lastRange = self.range        
            self.lastPoints = points
            self.lastPulses = pulses
                            
        return self.subsetColumns(self.lastPoints, colNames)
        
    def readPulsesForRange(self, colNames=None):
        """
        Reads the pulses for the current range. Returns a 1d array.

        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        if self.lastRange is None or self.range != self.lastRange:
            pulses, points = self.scanFile.readData(self.range.startPulse, 
                            self.range.endPulse)
            self.lastRange = self.range        
            self.lastPoints = points
            self.lastPulses = pulses
                            
        return self.subsetColumns(self.lastPulses, colNames)
        
    def getTotalNumberPulses(self):
        """
        We can assume that is there were waveforms present then
        the number of waveforms is equal to the number of pulses.
        Otherwise, no idea how to find out how many pulses so unsupported
        """
        if self.haveWave:
            return self.scanFile.numWaveRecords
        else:
            raise generic.LiDARFunctionUnsupported()

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        This driver does not support writing so ignore if reading,
        throw and error otherwise.
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return
        
        msg = 'riegl driver does not support update/creating'
        raise generic.LiDARWritingNotSupported(msg)
        
    def getHeader(self):
        """
        Riegl doesn't seem to have a header as such but we can 
        fake it by providing the info from _riegl.getFileInfo
        """
        if self.header is None:
            self.header = _riegl.getFileInfo(self.fname)
        return self.header
        
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]
    

class RieglFileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a Riegl file
    and makes it available as fields.
    The underlying C++ _riegl module does the hard work here
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)
                
        if not isRieglFile(fname):
            msg = 'not a riegl file'
            raise generic.LiDARFormatNotUnderstood(msg)

        self.header = _riegl.getFileInfo(fname)

    @staticmethod        
    def getDriverName():
        return 'riegl'

    @staticmethod
    def getHeaderTranslationDict():
        """
        Nothing yet - empty dict
        """
        return {}
