
"""
Classes that are passed to the user's function
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

import copy

class UserInfo(object):
    # equivalent to rios 'info'
    def __init__(self):
        self.pixGrid = None
        self.extent = None
        
    def setPixGrid(self, pixGrid):
        # take a copy so the user can't change it
        self.pixGrid = copy.copy(pixGrid)
        
    def getPixGrid(self):
        return self.pixGrid
        
    def setExtent(self, extent):
        # take a copy so the user can't change it
        self.extent = copy.copy(extent)
        
    def getExtent(self):
        return self.extent

class DataContainer(object):
    "UserInfo object plus instances of LidarData and ImageData"
    def __init__(self):
        self.info = UserInfo()

class LidarData(object):
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        self.extent = None
        
    def getPoints(self):
        "as a structured array"
        points = self.driver.readPointsForExtent()
        return points
        
    def getPulses(self):
        "as a structured array"
        pulses = self.driver.readPulsesForExtent()
        return pulses
        
    def getPulsesByBins(self):
        import numpy
        pulses = self.getPulses()
        idx = self.driver.lastPulses_Idx
        idxMask = self.driver.lastPulses_IdxMask 
        pulsesByBins = pulses[idx]
        return numpy.ma.array(pulsesByBins, mask=idxMask)
        
    def getPointsByPulse(self):
        import numpy
        points = self.getPoints()
        idx = self.driver.lastPoints_Idx
        idxMask = self.driver.lastPoints_IdxMask
        pointsByBins = points[idx]
        return numpy.ma.array(pointsByBins, mask=idxMask)
        
    def convertPointsTo2D(self, points):
        pass
        
    # For a particular pulse
    # TODO: should support a pulsearray with masking
    def getTransmitted(self, pulse):
        "as an integer array"
        return self.driver.readTransmitted(pulse)
        
    def setTransmitted(self, pulse, transmitted):
        "as an integer array"
        
    def getReceived(self, pulse):
        "as an integer array"
        return self.driver.readReceived(pulse)
        
    def setReceived(self, pulse, received):
        "as an integer array"
        
    def regridData(self, data):
        "tdb"
        
    def setPoints(self, pts):
        "as a structured array"
        self.driver.writePointsForExtent(points)
        
    def setPulses(self, pulses):
        "as a structured array"
        self.driver.writePulsesForExtent(pulses)
        
class ImageData(object):
    def __init__(self, mode, driver):
        self.mode = mode
        self.driver = driver
        
    def getData(self):
        "as 3d array"
        self.driver.getData()
        
    def setData(self, data):
        "as 3d array"    
        self.driver.setData(data)
            
            