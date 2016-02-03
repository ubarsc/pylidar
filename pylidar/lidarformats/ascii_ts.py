
"""
Driver for ASCII time sequential files

Driver Options
--------------

These are contained in the SUPPORTEDOPTIONS module level variable.

+-----------------------+-------------------------------------------+
| Name                  | Use                                       |
+=======================+===========================================+
| COL_TYPES             | A numpy style list of tuples defining     |
|                       | the data types of each column. Each tuple |
|                       | should have a name and a numpy dtype.     |
+-----------------------+-------------------------------------------+
| PULSE_COLS            | A list of fields which define the pulses. |
|                       | The values in the these columns will be   |
|                       | matched and where equal will be put into  |
|                       | one pulse and the other values into the   |
|                       | points for that pulse.                    |
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

import sys
import gzip
import copy
import numpy

from . import generic
from . import gridindexutils

SUPPORTEDOPTIONS = ('COL_TYPES', 'PULSE_COLS')
SEPARATOR = ','

class ASCIITSFile(generic.LiDARFile):
    """
    Driver for reading Riegl rxp files. Uses rivlib
    via the _riegl module.
    """
    def __init__(self, fname, mode, controls, userClass):
        generic.LiDARFile.__init__(self, fname, mode, controls, userClass)
        
        if mode != generic.READ:
            msg = 'ASCII TS driver is read only'
            raise generic.LiDARInvalidSetting(msg)

        if not fname.endswith('.dat.gz'):
            msg = 'only process *.dat.gz files at present'
            raise generic.LiDARFileException(msg)

        # check if the options are all valid. Good to check in case of typo.
        # hard to do this in C
        for key in userClass.lidarDriverOptions:
            if key not in SUPPORTEDOPTIONS:
                msg = '%s not a supported ASCII TS option' % repr(key)
                print(msg)
                raise generic.LiDARInvalidSetting(msg)

        # also check they are all present - we can't read the file otherwise
        for key in SUPPORTEDOPTIONS:
            if key not in userClass.lidarDriverOptions:
                msg = 'must provide %s driver option' % key
                print(msg)
                raise generic.LiDARInvalidSetting(msg)

        # save the types
        self.colTypes = userClass.lidarDriverOptions['COL_TYPES']
        # turn the pulse names into indices
        pulseCols = userClass.lidarDriverOptions['PULSE_COLS']
        self.pulseIdxs = []
        self.colDtype = []
        self.pulseDtype = []

        # might be a more efficient way of doing this
        # first process the pulseCols and build self.pulseDtype
        # plus self.pulseIdxs
        for key in pulseCols:
            found = False
            idx = 0
            for name, dt in self.colTypes:
                if name == key:
                    found = True
                    self.pulseIdxs.append(idx)
                    self.pulseDtype.append((name, dt))
                    break
                idx += 1

            if not found:
                msg = 'Cannot find pulse column %s in COL_TYPES' % key
                raise generic.LiDARInvalidSetting(msg)

        # append the fields that refer to the points
        self.pulseDtype.append(('NUMBER_OF_RETURNS', numpy.uint8))
        self.pulseDtype.append(('PTS_START_IDX',  numpy.uint64))

        # now go through and build self.pointDtype
        # and self.pointIdxs
        self.pointDtype = []
        self.pointIdxs = []
        idx = 0
        for name, dt in self.colTypes:
            if idx not in self.pulseIdxs:
                self.pointDtype.append((name, dt))
                self.pointIdxs.append(idx)
            idx += 1

        self.fh = gzip.open(fname, 'r')
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None
        self.finished = False
            
    @staticmethod        
    def getDriverName():
        return 'ASCII TS'

    def close(self):
        self.fh.close()
        self.fh = None
        self.range = None
        self.lastRange = None
        self.lastPoints = None
        self.lastPulses = None

    def hasSpatialIndex(self):
        """
        ASCII files aren't spatially indexed
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
        return not self.finished

    def readData(self):
        """
        Internal method. Reads all the points and pulses
        for the current pulse range.
        """
        # we know how many pulses we have so we can create that 
        # array
        nPulses = self.range.endPulse - self.range.startPulse
        pulses = numpy.empty(nPulses, dtype=self.pulseDtype)
        points = None
        lastPulseDict = {}
        for idx in self.pulseIdxs:
            lastPulseDict[idx] = ''

        pulseCount = 0
        pointCount = 0
        while pulseCount < nPulses:
            line = self.fh.readline()
            if sys.version_info[0] >= 3:
                line = line.decode()
            dataArr = line.strip('\r\n').split(SEPARATOR)

            if len(dataArr) <= 1:
                # seems to have single element when end of file....
                self.finished = True
                break

            # first pass - determine if new pulse or not
            samePulse = True
            for idx in self.pulseIdxs:
                if dataArr[idx] != lastPulseDict[idx]:
                    samePulse = False
                    break

            #print(pulseCount, pointCount, samePulse, dataArr)
            if not samePulse:
                # append to our array of pulses
                for idx in self.pulseIdxs:
                    name, dt = self.colTypes[idx]
                    pulses[pulseCount][name] = dt(dataArr[idx])
                    # update our dict so we can detect next one
                    lastPulseDict[idx] = dataArr[idx]

                # refering to the points
                pulses[pulseCount]['NUMBER_OF_RETURNS'] = 0
                pulses[pulseCount]['PTS_START_IDX'] = pointCount

                pulseCount += 1
                        
            # right, do points
            if points is None:
                points = numpy.empty(1, dtype=self.pointDtype)
            else:
                newpoint = numpy.empty(1, dtype=self.pointDtype)
                points = numpy.append(points, newpoint)

            for idx in self.pointIdxs:
                name, dt = self.colTypes[idx]
                points[pointCount][name] = dt(dataArr[idx])

            # update pulse
            pulses[pulseCount-1]['NUMBER_OF_RETURNS'] += 1

            pointCount += 1

        return pulses, points


    def readPointsForRange(self, colNames=None):
        """
        Reads the points for the current range. Returns a 1d array.
        
        Returns an empty array if range is outside of the current file.

        colNames can be a list of column names to return. By default
        all columns are returned.
        """
        if self.lastRange is None or self.range != self.lastRange:
            pulses, points = self.readData()
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
            pulses, points = self.readData()
            self.lastRange = self.range        
            self.lastPoints = points
            self.lastPulses = pulses
                            
        return self.subsetColumns(self.lastPulses, colNames)

    def readPointsByPulse(self, colNames=None):
        """
        Read a 3d structured masked array containing the points
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

    def getTotalNumberPulses(self):
        """
        No idea how to find this out...
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
        
        msg = 'ASCII TS driver does not support update/creating'
        raise generic.LiDARWritingNotSupported(msg)

    def getHeader(self):
        """
        ASCII files have no header
        """
        return {}
        
    def getHeaderValue(self, name):
        """
        Just extract the one value and return it
        """
        return self.getHeader()[name]

    def readTransmitted(self):
        """
        ASCII (AFAIK) doesn't support transmitted
        """
        return None
        
    def readReceived(self):
        """
        ASCII (AFAIK) doesn't support received
        """
        return None

    def readWaveformInfo(self):
        """
        ASCII (AFAIK) doesn't support waveforms
        """
        return None
