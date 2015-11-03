
"""
Driver for riegl rxp files
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

import copy
import numpy

from . import generic
from . import _riegl

class RieglFile(generic.LiDARFile):
    """
    Driver for reading Riegl rxp files. Uses rivlib
    via the _riegl module.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)

        # The riegl supplied functions only return an error when 
        # we actually start reading but PyLidar needs to know straight
        # away. Workaround is to read the start of the file and see
        # if it contains 'Riegl'. Normally at byte position 9 but we check
        # any of the first 32 bytes.
        fh = open(fname, 'rb')
        data = fh.read(32)
        fh.close()
        
        if data.find(b'Riegl') == -1:
            msg = 'not a riegl file'
            raise generic.LiDARFileException(msg)
        
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.scanFile = _riegl.ScanFile(fname)
                          
    @staticmethod        
    def getDriverName():
        return 'riegl'
        
    def close(self):
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.scanFile = None
        
    def readPointsByPulse(self):     
        """
        Read a 3d structured masked array containing the points
        for each pulse.
        """
        raise NotImplementedError()
        
    def readWaveformInfo(self):
        """
        2d structured masked array containing information
        about the waveforms.
        """
        raise NotImplementedError()
        
    def readTransmitted(self):
        """
        Read the transmitted waveform for all pulses
        returns a 3d masked array. 
        """
        raise NotImplementedError()
        
    def readReceived(self):
        """
        Read the received waveform for all pulses
        returns a 2d masked array
        """
        raise NotImplementedError()

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
        No idea how to find out how many pulses so unsupported
        """
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
        
    @staticmethod
    def subsetColumns(array, colNames):
        """
        Internal method. Subsets the given column names from the array and
        returns it. colNames can be either a string or a sequence of column
        names. If None the input array is returned.
        """
        if colNames is not None:
            if isinstance(colNames, str):
                array = array[colNames]
            else:
                # assume a sequence. For some reason numpy
                # doesn't like a tuple here
                colNames = list(colNames)
                
                # need to check that all the named columns
                # actually exist in the structured array.
                # Numpy gives no error/warning if they do not
                # just simply ignores ones that don't exist.
                existingNames = array.dtype.fields.keys()
                for col in colNames:
                    if col not in existingNames:
                        msg = 'column %s does not exist for this format' % col
                        raise generic.LiDARArrayColumnError(msg)
                
                # have to do a copy to avoid numpy warning
                # that updating returned array will break in future
                # numpy release.
                array = array[colNames].copy()
            
        return array
        