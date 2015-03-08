
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

from __future__ import print_function, division

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
        if self.driver.controls.spatialProcessing:
            points = self.driver.readPointsForExtent()
        else:
            points = self.driver.readPointsForRange()
        return points
        
    def getPulses(self):
        "as a structured array"
        if self.driver.controls.spatialProcessing:
            pulses = self.driver.readPulsesForExtent()
        else:
            pulses = self.driver.readPulsesForRange()
        return pulses
        
    def getPulsesByBins(self, extent=None):
        # TODO: move into driver
        # maybe driver should return object 
        # with other info from getPulses()
        import numpy
        pulses = self.getPulses()
        idx = self.driver.lastPulses_Idx
        idxMask = self.driver.lastPulses_IdxMask 
        pulsesByBins = pulses[idx]
        return numpy.ma.array(pulsesByBins, mask=idxMask)
        
    def getPointsByBins(self, extent=None):
        # have to spatially index the points
        import numpy
        points = self.getPoints()
        extent = self.driver.lastExtent
        nrows = int((extent.yMax - extent.yMin) / extent.binSize)
        ncols = int((extent.xMax - extent.xMin) / extent.binSize)
        sortedbins, idx, cnt = self.driver.CreateSpatialIndex(
                points['Y'], points['X'], extent.binSize, extent.yMax,
                extent.xMin, nrows, ncols)
        # TODO: don't really want the bool array returned - need
        # to make it optional
        nOut = len(points)
        pts_bool, pts_idx, pts_idx_mask = self.driver.convertSPDIdxToReadIdxAndMaskInfo(
                                idx, cnt, nOut)
                                
        sortedPoints = points[sortedbins]
        
        pointsByBins = sortedPoints[pts_idx]
        return numpy.ma.array(pointsByBins, mask=pts_idx_mask)

    def getPointsByPulse(self):
        import numpy
        points = self.getPoints()
        idx = self.driver.lastPoints_Idx
        idxMask = self.driver.lastPoints_IdxMask
        
        pointsByPulse = points[idx]
        return numpy.ma.array(pointsByPulse, mask=idxMask)
        
    def getTransmitted(self):
        "as a masked 2d integer array"
        return self.driver.readTransmitted()
        
    def setTransmitted(self, transmitted):
        "as a masked 2d integer array"
        
    def getReceived(self):
        "as an integer array"
        return self.driver.readReceived()
        
    def setReceived(self, received):
        "as an integer array"
        
    def setPoints(self, points):
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
