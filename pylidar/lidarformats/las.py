"""
Driver for .las files. Uses lastools (https://github.com/LAStools/LAStools).

Driver Options
--------------

These are contained in the SUPPORTEDOPTIONS module level variable.

+-----------------------+-------------------------------------------+
| Name                  | Use                                       |
+=======================+===========================================+
| BUILD_PULSES          | a boolean. If set to true (the default)   |
|                       | pylidar attempts to build pulses assuming |
|                       | that data is in time sequential order. If |
|                       | false, a 'fake' pulse is created for each |
|                       | point.                                    |
+-----------------------+-------------------------------------------+

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
from . import _las
from . import gridindexutils

SUPPORTEDOPTIONS = _las.getSupportedOptions()

def isLasFile(fname):
    """
    Helper function that looks at the start of the file
    to determine if it is a las file or not
    """
    # The las library reads the whole file before failing
    # and prints a whole lot of stuff to stdout.
    # Easier just to see if file starts with 'LASF'
    fh = open(fname, 'rb')
    data = fh.read(4)
    fh.close()
        
    if data != b'LASF':
        return False
    else:
        return True

class LasFile(generic.LiDARFile):
    """
    Reader for .las files.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)

        if not isLasFile(fname):
            msg = 'not a las file'
            raise generic.LiDARFileException(msg)
        
        try:
            self.lasFile = _las.LasFile(fname, userClass.lidarDriverOptions)
        except:
            msg = 'cannot open as las file'
            raise generic.LiDARFileException(msg)
            
        print('opened las ok')

    @staticmethod        
    def getDriverName():
        return 'las'
        
    def close(self):
        self.lasFile = None
        
    def hasSpatialIndex(self):
        """
        TODO: check
        """
        return False
    
    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        # TODO:
        pass

    def readPointsForRange(self, colNames=None):
        # TODO:
        pass
        
    def readPulsesForRange(self, colNames=None):
        # TODO:
        pass        
        
    def readWaveformInfo(self):
        """
        3d structured masked array containing information
        about the waveforms.
        """
        # TODO:
        pass        
        
    def readTransmitted(self):
        """
        Riegl (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        # TODO:
        pass        
        
    def getTotalNumberPulses(self):
        # TODO:
        pass
        
    def writeData(self, pulses=None, points=None, transmitted=None, 
                received=None, waveformInfo=None):
        """
        This driver does not support writing so ignore if reading,
        throw and error otherwise.
        """
        if self.mode == generic.READ:
            # the processor always calls this so if a reading driver just ignore
            return
        
        msg = 'las driver does not support update/creating'
        raise generic.LiDARWritingNotSupported(msg)
                
    def getHeader(self):
        # TODO:
        return {}
        
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]
            