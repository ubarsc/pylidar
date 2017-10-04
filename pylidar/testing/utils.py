
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
import os
import sys
import copy
import json
import shutil
import hashlib
import tarfile
import subprocess
import numpy
from pylidar import lidarprocessor
from rios import cuiprogress
from rios.parallel.jobmanager import find_executable

TESTSUITE_VERSION = 9
"""
Version of the test suite. Increment each change.
Used to ensure the tarfile matches what we expect.
"""

TESTDATA_DIR = 'testdata'
"subdirectory within the tar file with the data"

NEWDATA_DIR = 'newdata'
"""
subdirectory that will be created where the tar file is extracted
that will contain the 'new' files which will be compared with those
in TESTDATA_DIR.
"""

VERSION_FILE = 'version.txt'
"name of the file containing the version information in the tar file"

# maybe these should be in lidarprocessor also?
ARRAY_TYPE_TRANSMITTED = 100
ARRAY_TYPE_RECEIVED = 101

# path to gdalchksum.py so we can run it on Windows
GDALCHKSUM = find_executable('gdalchksum.py')
if GDALCHKSUM is None:
    raise IOError('Cannot find gdalchksum.py in $PATH')

class TestingError(Exception):
    "Base class for testing Exceptions"

class TestingVersionError(TestingError):
    "Was a mismatch in versions"

class TestingDataMismatch(TestingError):
    "Data does not match between expected and newly calculated"

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
    header = {}

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

        names = dict1.keys()
        if names != dict2.keys():
            return False

        for name in names:
             if dict1[name] != dict2[name]:
                return False
        return True

    @staticmethod
    def cmpDictWithError(olddict, newdict, atype):
        """
        Internal method. Compares 2 dictionaries
        and raises an Exception with a message if they don't match.

        atype should be point|pulse|waveform|header and is included in the
        error message.
        """
        if olddict is None and newdict is None:
            return True
        if olddict is not None and newdict is None: 
            msg = '%s present one the original file but not in the new one'
            msg = msg % atype
            raise TestingDataMismatch(msg)
        if olddict is None and newdict is not None:
            msg = '%s present one the mew file but not in the original one'
            msg = msg % atype
            raise TestingDataMismatch(msg)

        names = olddict.keys()
        if names != newdict.keys():
            msg = "for %s the column names don't match. Original: %s New: %s"
            msg = msg % (atype, ','.join(names), ','.join(newdict.keys()))
            raise TestingDataMismatch(msg)

        for name in names:
             if newdict[name] != olddict[name]:
                msg = 'for %s there is a data mismatch on column %s'
                msg = msg % (atype, name)
                raise TestingDataMismatch(msg)

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
            and self.receivedSize == other.receivedSize
            and self.header == other.header)

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
            or self.receivedSize != other.receivedSize
            or self.header != other.header)

    def doCheck(self, other):
        """
        Similar to __eq__, but raises an exception with a useful message 
        Expects self to be the 'old' file info.
        """
        if self.pointSize != other.pointSize:
            msg = 'old file has a different number of points to the new file'
            raise TestingDataMismatch(msg)

        if self.pulseSize != other.pulseSize:
            msg = 'old file has a different number of pulses to the new file'
            raise TestingDataMismatch(msg)

        if self.waveformSize != other.waveformSize:
            msg = 'old file has a different number of waveforms to the new file'
            raise TestingDataMismatch(msg)

        if self.transmittedSize != other.transmittedSize:
            msg = 'old file has a different number of transmitted to the new file'
            raise TestingDataMismatch(msg)

        if self.transmittedChecksum != other.transmittedChecksum:
            msg = "transmitted checksums don't match"
            raise TestingDataMismatch(msg)

        if self.receivedSize != other.receivedSize:
            msg = 'old file has a different number of received to the new file'
            raise TestingDataMismatch(msg)

        if self.receivedChecksum != other.receivedChecksum:
            msg = "received checksums don't match"
            raise TestingDataMismatch(msg)

        self.cmpDictWithError(self.pointChecksums, other.pointChecksums, 'point')
        self.cmpDictWithError(self.pulseChecksums, other.pulseChecksums, 'pulse')
        self.cmpDictWithError(self.waveformChecksums, other.waveformChecksums, 
                'waveform')
        self.cmpDictWithError(self.waveformChecksums, other.waveformChecksums, 
                'header')
        
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

    def setHeader(self, header):
        """
        Sets the header so we can compare with another header later
        """
        self.header = copy.copy(header)

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

        if self.transmittedChecksum is not None:
            self.transmittedChecksum = self.transmittedChecksum.hexdigest()
        if self.receivedChecksum is not None:
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

    if data.info.isFirstBlock():
        header = data.input.getHeader()
        otherargs.checksum.setHeader(header)

def calculateCheckSum(infile, windowSize=None):
    """
    Returns a Checksum instance for the given file
    """
    print('Calculating LiDAR Checksum...')
    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile(infile, lidarprocessor.READ)

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.checksum = Checksum()

    controls = lidarprocessor.Controls()
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    controls.setMessageHandler(lidarprocessor.silentMessageFn)
    if windowSize is not None:
        controls.setWindowSize(windowSize)

    lidarprocessor.doProcessing(pylidarChecksum, dataFiles, otherArgs=otherArgs,
            controls=controls)

    # as a last step, calculate the digests
    otherArgs.checksum.convertToDigests()

    return otherArgs.checksum

def compareLiDARFiles(oldfile, newfile, windowSize=None):
    """
    Compares 2 LiDAR files and raises and exception with information 
    about the differences if they do not match.
    """
    oldChecksum = calculateCheckSum(oldfile, windowSize)
    newChecksum = calculateCheckSum(newfile, windowSize)
    oldChecksum.doCheck(newChecksum)
    print('LiDAR files check ok')

def compareImageFiles(oldfile, newfile):
    """
    Compares two image files using gdalchksum.py
    """
    # be explicit when starting gdalchksum.py since this
    # is needed on Windows. Doesn't hurt on other platforms
    # tho.
    oldChecksum = subprocess.check_output([sys.executable, 
                    GDALCHKSUM, oldfile])
    newChecksum = subprocess.check_output([sys.executable, 
                    GDALCHKSUM, newfile])
    if oldChecksum != newChecksum:
        msg = 'image checksums do not match'
        raise TestingDataMismatch(msg)
    print('Image files check ok')

def compareNumpyFiles(oldfile, newfile):
    """
    Compares 2 data files saved in the numpy.save format
    """
    olddata = numpy.load(oldfile)
    olddata_nan = numpy.isnan(olddata)
    # == operator does weird things with NaNs
    olddata = olddata[~olddata_nan]

    newdata = numpy.load(newfile)
    newdata_nan = numpy.isnan(newdata)
    newdata = newdata[~newdata_nan]

    if (olddata.shape != newdata.shape or 
            olddata_nan.shape != newdata_nan.shape):
        msg = 'numpy data is different size'
        raise TestingDataMismatch(msg)

    if not (olddata_nan == newdata_nan).all():
        msg = 'numpy nan data does not match'
        raise TestingDataMismatch(msg)

    # could potentially do these in chunks
    # or set mmap_mode in numpy.load to reduce memory usage
    if not (olddata == newdata).all():
        msg = 'numpy data does not match'
        raise TestingDataMismatch(msg)
    print('numpy data checks ok')

def extractTarFile(tarFile, pathToUse='.', doVersionCheck=True):
    """
    Extracts the tarFile to the given path and checks the version matches
    what was expected. 

    Returns the path to where the data files are (ie inside the extracted tarfile)
    and the path to where to create the new data files for comparison and the list
    of tests in the tar file (there should be a module for each of these)
    """
    outDataDir = os.path.join(pathToUse, TESTDATA_DIR)
    if os.path.isdir(outDataDir):
        # remove it
        shutil.rmtree(outDataDir, ignore_errors=True)

    # extract everything
    tar = tarfile.open(tarFile)
    tar.extractall(pathToUse)
    tar.close()

    # check the version 
    versionPath = os.path.join(outDataDir, VERSION_FILE)
    data = open(versionPath).readline()

    dataDict = json.loads(data)

    if doVersionCheck and dataDict['version'] != TESTSUITE_VERSION:
        msg = "Version is match. Expected %d but tarfile is version %d"
        msg = msg % (TESTSUITE_VERSION, dataDict['version'])
        raise TestingVersionError(msg)

    tests = dataDict['tests']

    # create the NEWDATA_DIR
    newDataDir = os.path.join(pathToUse, NEWDATA_DIR)
    if os.path.isdir(newDataDir):
        # remove it
        shutil.rmtree(newDataDir, ignore_errors=True)

    # create it
    os.mkdir(newDataDir)

    return outDataDir, newDataDir, tests
