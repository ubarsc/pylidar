
"""
Base class for LiDAR format reader/writers
"""
# This file is part of PyLidar
# Copyright (C) 2015 John Armston, Neil Flood and Sam Gillingham
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

from .. import basedriver

READ = 0
UPDATE = 1
CREATE = 2

class LiDARFileException(Exception):
    "Base class for LiDAR format reader/writers"
    
class LiDARFormatNotUnderstood(LiDARFileException):
    "Raised when driver cannot open file"
    
class LiDARFormatDriverNotFound(LiDARFileException):
    "None of the drivers can open the file"
    
class LiDARInvalidData(LiDARFileException):
    "Something is wrong with the data read or given"
    
class LiDARInvalidSetting(LiDARFileException):
    "Setting does not make sense"
    
class PulseRange(object):
    """
    Class for setting the range of pulses to read/write
    for non spatial mode.
    Note: range does not include endPulse
    """
    def __init__(self, startPulse, endPulse):
        self.startPulse = startPulse
        self.endPulse = endPulse
        
    def __eq__(self, other):
        return (self.startPulse == other.startPulse and 
                self.endPulse == other.endPulse)
                
    def __ne__(self, other):
        return (self.startPulse != other.startPulse or
                self.endPulse != other.endPulse)

class LiDARFile(basedriver.Driver):
    """
    Base class for all LiDAR Format reader/writers
    """
    def __init__(self, fname, mode, controls, userClass):
        basedriver.Driver.__init__(self, fname, mode, controls, userClass)
        
    def getDriverName(self):
        """
        Return name of driver
        """
        raise NotImplementedError()
        
    def readPointsForExtent(self):
        """
        Read all the points within the given extent
        as 1d strcutured array
        """
        raise NotImplementedError()
        
    def readPulsesForExtent(self):
        """
        Read all the pulses within the given extent
        as 1d strcutured array
        """
        raise NotImplementedError()
        
    def readTransmitted(self):
        """
        Read the transmitted waveform for all pulses
        returns a 2d masked array
        """
        raise NotImplementedError()
        
    def readReceived(self):
        """
        Read the received waveform for all pulses
        returns a 2d masked array
        """
        raise NotImplementedError()
        
    def writePointsForExtent(self, points):
        raise NotImplementedError()
        
    def writePulsesForExtent(self, pulses):
        raise NotImplementedError()
        
    def writeTransmitted(self, pulses, transmitted):
        raise NotImplementedError()
        
    def writeReceived(self, pulses, received):
        raise NotImplementedError()
        
    def hasSpatialIndex(self):
        """
        Returns True if file has a spatial index defined
        """
        raise NotImplementedError()
        
    # see below for no spatial index
    def setPulseRange(self, pulseRange):
        """
        Sets the PulseRange object to use for non spatial
        reads/writes.
        """
        raise NotImplementedError()
    
    def readPointsForRange(self):
        raise NotImplementedError()
        
    def readPulsesForRange(self):
        raise NotImplementedError()
        
    def getTotalNumberPulses(self):
        raise NotImplementedError()
        
    def writePoints(self, points):
        raise NotImplementedError()
        
    def writePulses(self, pulses):
        raise NotImplementedError()
        
    def close(self):
        raise NotImplementedError()


def getReaderForLiDARFile(fname, mode, controls, userClass):
    """
    Returns an instance of a LiDAR format
    reader/writer or raises an exception if none
    found for the file.
    """
    # try each subclass
    for cls in LiDARFile.__subclasses__():
        #print('trying', cls)
        try:
            # attempt to create it
            inst = cls(fname, mode, controls, userClass)
            # worked - return it
            return inst
        except LiDARFileException:
            # failed - onto the next one
            pass
    # none worked
    msg = 'Cannot open LiDAR file %s' % fname
    raise LiDARFormatDriverNotFound(msg)

