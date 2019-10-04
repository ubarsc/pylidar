
"""
Driver for Riegl .rdbx files

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
import json
import numpy

from . import generic
# Fail slightly less drastically when running from ReadTheDocs
if os.getenv('READTHEDOCS', default='False') != 'True':
    from . import _rieglrdb

from . import gridindexutils

RDB_MAGIC = b'RIEGL LMS RDB 2 POINTCLOUD FILE'

def isRieglRDBFile(fname):
    """
    Helper function that looks at the start of the file
    to determine if it is a riegl RDB file or not.
    Thankfully this is easy since RDB_MAGIC is at the start
    of the file.
    """
    fh = open(fname, 'rb')
    data = fh.read(len(RDB_MAGIC))
    fh.close()
        
    return data == RDB_MAGIC

class RieglRDBFile(generic.LiDARFile):
    """
    Driver for reading Riegl RDB files. Uses rdblib
    via the _rieglrdb module.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)
        
        if mode != generic.READ:
            msg = 'Riegl RDB driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        if not isRieglRDBFile(fname):
            msg = 'not a riegl RDB file'
            raise generic.LiDARFileException(msg)
        
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        # cache header so we can do a json.loads on everything
        self.header = None
        self.rdbFile = _rieglrdb.RDBFile(fname)
        
    @staticmethod        
    def getDriverName():
        return 'riegl RDB'

    @staticmethod
    def getTranslationDict(arrayType):
        """
        Translation dictionary between formats
        """
        dict = {}
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
        self.header = None
        self.rdbFile = None
        
    def getTotalNumberPulses(self):
        """
        No idea how to find out how many pulses so unsupported
        """
        raise generic.LiDARFunctionUnsupported()

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
            
    def readWaveformInfo(self):
        return None
    
    def readTransmitted(self):
        """
        Riegl RDB (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        """
        Riegl RDB (AFAIK) doesn't support received
        """
        return None

    def hasSpatialIndex(self):
        """
        Riegl RDB files aren't spatially indexed (?)
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
        return not self.rdbFile.finished
    
    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        if self.lastRange is None or self.range != self.lastRange:
            pulses, points = self.rdbFile.readData(self.range.startPulse, 
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
            pulses, points = self.rdbFile.readData(self.range.startPulse, 
                            self.range.endPulse)
            self.lastRange = self.range        
            self.lastPoints = points
            self.lastPulses = pulses
                            
        return self.subsetColumns(self.lastPulses, colNames)

    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        This driver does not support writing so ignore if reading,
        throw and error otherwise.
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return
        
        msg = 'riegl RDB driver does not support update/creating'
        raise generic.LiDARWritingNotSupported(msg)
        
    def getHeader(self):
        """
        Get header from C++ if we haven't already. Most of it appears
        to be JSON strings so we attempt to decode them back into Python
        objects. We don't seem to be able to do this from C++ directly.
        """
        if self.header is None:
            header = self.rdbFile.header
            self.header = {}
            for key in header:
                try:
                    self.header[key] = json.loads(header[key])
                except (json.decoder.JSONDecodeError, TypeError):
                    # just copy value across - not JSON
                    self.header[key] = header[key]
            
        return self.header
        
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]
     
# There is a lot of stuff in the header. Just pull out some fields
# that look like they might be interesting
INFO_KEYS = ('riegl.geo_tag', 'riegl.time_base', 'riegl.device')
     
class RieglRDBFileInfo(generic.LiDARFileInfo):
    """
    Class that gets information about a Riegl file
    and makes it available as fields.
    The underlying C++ _riegl module does the hard work here
    """
    def __init__(self, fname):
        generic.LiDARFileInfo.__init__(self, fname)
                
        if not isRieglRDBFile(fname):
            msg = 'not a riegl RDB file'
            raise generic.LiDARFormatNotUnderstood(msg)

        rdbFile = _rieglrdb.RDBFile(fname)
        self.header = {}
        for name in INFO_KEYS:
            if name in rdbFile.header:
                self.header[name] = rdbFile.header[name]

    @staticmethod        
    def getDriverName():
        return 'riegl RDB'

    @staticmethod
    def getHeaderTranslationDict():
        """
        Nothing yet - empty dict
        """
        return {}
