
"""
General testing utility functions.
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

import hashlib
from pylidar import lidarprocessor

# maybe these should be in lidarprocessor also?
ARRAY_TYPE_TRANSMITTED = 100
ARRAY_TYPE_RECEIVED = 101

class Checksum(object):
    """
    Holds the separate checksum dictionaries for points, pulses etc.
    """
    pointChecksums = None
    pointSize = 0
    pulseChecksums = None
    pulseSize = 0
    waveformChecksums = None
    waveformSize = 0
    transmittedChecksum = None
    transmittedSize = 0
    receivedChecksum = None
    receivedSize = 0

    @staticmethod
    def cmpDict(dict1, dict2):
        """
        Internal method. Compares 2 dictionaries
        and returns True if equal.
        """
        if dict1 is None and dict2 is None:
            return True
        if (dict1 is not None and dict2 is None) or (dict1 is None and dict2 is not None):
            return False

        names = dict1.dtype.names
        if names != dict2.dtype.names:
            return False

        for name in names:
             if dict1[name] != dict2[name]:
                return False
        return True

    def __eq__(self, other):
        return (self.cmpDict(self.pointChecksums, other.pointChecksums) 
            and self.pointSize == other.pointSize
            and self.cmpDict(self.pulseChecksums, other.pulseChecksums)
            and self.pulseSize == other.pulseSize
            and self.cmpDict(self.waveformChecksums, other.waveformChecksums)
            and self.waveformSize == other.waveformSize
            and self.transmittedChecksum == other.transmittedChecksum
            and self.transmittedSize == other.transmittedSize
            and self.receivedChecksum == other.receivedChecksum
            and self.receivedSize == other.receivedSize)

    def __ne__(self, other):
        return (not self.cmpDict(self.pointChecksums, other.pointChecksums) 
            or self.pointSize != other.pointSize
            or not self.cmpDict(self.pulseChecksums, other.pulseChecksums)
            or self.pulseSize != other.pulseSize
            or not self.cmpDict(self.waveformChecksums, other.waveformChecksums)
            or self.waveformSize != other.waveformSize
            or self.transmittedChecksum != other.transmittedChecksum
            or self.transmittedSize != other.transmittedSize
            or self.receivedChecksum != other.receivedChecksum
            or self.receivedSize != other.receivedSize)

    def __str__(self):
        s = 'pointSize:%d,pulseSize:%d,waveInfoSize:%d,transmittedSize:%d,receivedSize:%d' % (
            self.pointSize, self.pulseSize, self.waveformSize, 
            self.transmittedSize, self.receivedSize)
        # now add the digests
        if self.pointChecksums is not None:
            s = s + '\nPoint Checksums:\n'
            for name in self.pointChecksums:
                s = s + name + ' = ' + self.pointChecksums[name] + '\n'

        if self.pulseChecksums is not None:
            s = s + '\nPulse Checksums:\n'
            for name in self.pulseChecksums:
                s = s + name + ' = ' + self.pulseChecksums[name] + '\n'

        if self.waveformChecksums is not None:
            s = s + '\nWaveform Checksums:\n'
            for name in self.waveformChecksums:
                s = s + name + ' = ' + self.waveformChecksums[name] + '\n'

        if self.transmittedChecksum is not None:
            s = s + '\nTransmitted Checksum = ' + self.transmittedChecksum + '\n'

        if self.receivedChecksum is not None:
            s = s + '\nReceived Checksum = ' + self.receivedChecksum + '\n'

        return s

    @staticmethod
    def initDict(names):
        """
        Internal method. Returns a new dict with an entry
        for each name set to hashlib.md5()
        """
        ckdict = {}
        for name in names:
            ckdict[name] = hashlib.md5()
        return ckdict

    def updateChecksum(self, array, arrayType):
        """
        Updates the given checksum dictionary of md5 objects
        with the data in array. arrayType is one of the lidarprocessor
        ARRAY_TYPE_* constants.
        """
        if array is None:
            return None

        names = array.dtype.names
        if arrayType == lidarprocessor.ARRAY_TYPE_POINTS:
            if self.pointChecksums is None:
                self.pointChecksums = self.initDict(names)

            ckdict = self.pointChecksums
            self.pointSize += array.size
        elif arrayType == lidarprocessor.ARRAY_TYPE_PULSES:
            if self.pulseChecksums is None:
                self.pulseChecksums = self.initDict(names)

            ckdict = self.pulseChecksums
            self.pulseSize += array.size
        elif arrayType == lidarprocessor.ARRAY_TYPE_WAVEFORMS:
            if self.waveformChecksums is None:
                self.waveformChecksums = self.initDict(names)

            ckdict = self.waveformChecksums
            self.waveformSize += array.size

        elif arrayType == ARRAY_TYPE_TRANSMITTED:
            if self.transmittedChecksum is None:
                self.transmittedChecksum = hashlib.md5()

            ckdict = self.transmittedChecksum 
            self.transmittedSize += array.size

        elif arrayType == ARRAY_TYPE_RECEIVED:
            if self.receivedChecksum is None:
                self.receivedChecksum = hashlib.md5()

            ckdict = self.receivedChecksum 
            self.receivedSize += array.size

        if names is None:
            # transmitted or received
            ckdict.update(array)
        else:
            for name in names:
                # get ndarray is not C-contiguous errors if we don't copy
                data = array[name].copy()
                ckdict[name].update(data)

        return ckdict

    @staticmethod
    def convertDictToDigests(ckdict):
        """
        Internal method. Convert from md5 objects to hexdigests
        """
        if ckdict is not None:
            for name in ckdict.keys():
                ckdict[name] = ckdict[name].hexdigest()

    def convertToDigests(self):
        """
        Converts all the md5 objects in the dictionaries to
        digests for easier comparisons
        """
        self.convertDictToDigests(self.pulseChecksums)
        self.convertDictToDigests(self.pointChecksums)
        self.convertDictToDigests(self.waveformChecksums)
        self.transmittedChecksum = self.transmittedChecksum.hexdigest()
        self.receivedChecksum = self.receivedChecksum.hexdigest()

def pylidarChecksum(data, otherargs):
    """
    Internal method. Called by calculateCheckSum via
    lidarprocessor.
    """
    pulses = data.input.getPulses()
    otherargs.checksum.updateChecksum(pulses, lidarprocessor.ARRAY_TYPE_PULSES)

    points = data.input.getPulses()
    otherargs.checksum.updateChecksum(points, lidarprocessor.ARRAY_TYPE_POINTS)

    waveformInfo = data.input.getWaveformInfo()
    otherargs.checksum.updateChecksum(waveformInfo, lidarprocessor.ARRAY_TYPE_WAVEFORMS)
 
    transmitted = data.input.getTransmitted()   
    otherargs.checksum.updateChecksum(transmitted, ARRAY_TYPE_TRANSMITTED)

    received = data.input.getReceived()   
    otherargs.checksum.updateChecksum(received, ARRAY_TYPE_RECEIVED)

def calculateCheckSum(infile):
    """
    Returns a Checksum instance for the given file
    """
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.checksum = Checksum()

    lidarprocessor.doProcessing(pylidarChecksum, dataFiles, otherArgs=otherArgs)

    # as a last step, calculate the digests
    otherArgs.checksum.convertToDigests()

    return otherArgs.checksum
