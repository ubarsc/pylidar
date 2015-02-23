
"""
SPD V3
"""
import copy
import numpy
import h5py
from numba import jit
from rios import pixelgrid
from . import generic

@jit
def convertIdxBool(start_idx_array, count_array, out):
    """
    Convert SPD's default regular spatial index of pulse offsets and pulse counts
    per bin into a single boolean array. It is assumed that out is already
    set to all False

    Written to use numba - numpy version was very slow. Anyone any ideas
    on how to do this quickly in normal numpy?
    """
    for inidx in range(start_idx_array.shape[0]):
        cnt = count_array[inidx]
        startidx = start_idx_array[inidx]
        for i in range(cnt):
            # seems a strange bug in numba/llvm where the
            # result of this add gets promoted to a double
            # so cast it back
            outidx = int(startidx + i)
            out[outidx] = True
    # I believe that this calculation we have just done is also the perfect
    # place to assemble some sort of array showing which bin each pulse belongs with,
    # so it might be that we should return two things from this function. I don't yet 
    # know what such an array would look like, though....

class SPDV3File(generic.LiDARFile):
    def __init__(self, fname, mode, controls):
        generic.LiDARFile.__init__(self, fname, mode, controls)
    
        # convert mode into h5py mode string
        if mode == generic.READ:
            h5py_mode = 'r'
        elif mode == generic.UPDATE:
            h5py_mode = 'r+'
        elif mode == generic.CREATE:
            h5py_mode == 'w'
        else:
            raise ValueError('Unknown value for mode parameter')
    
        # attempt to open the file
        try:
            self.fileHandle = h5py.File(fname, h5py_mode)
        except OSError as err:
            # always seems to through an OSError
            raise generic.LiDARFormatNotUnderstood(str(err))
            
        # check that it is indeed the right version
        headerKeys = self.fileHandle['HEADER'].keys()
        if (not 'VERSION_MAJOR_SPD' in headerKeys or 
                    not 'VERSION_MINOR_SPD' in headerKeys):
            msg = "File appears not to be SPD"
            raise generic.LiDARFormatNotUnderstood(msg)
        elif self.fileHandle['HEADER']['VERSION_MAJOR_SPD'][0] != 2:
            msg = "File seems to be wrong version for this driver"
            raise generic.LiDARFormatNotUnderstood(msg)

        # read in the bits I need            
        if mode == generic.READ:
            self.si_cnt = self.fileHandle['INDEX']['PLS_PER_BIN'][...]
            self.si_idx = self.fileHandle['INDEX']['BIN_OFFSETS'][...]
            self.si_binSize = self.fileHandle['HEADER']['BIN_SIZE'][0]
            self.si_xMin = self.fileHandle['HEADER']['X_MIN'][0]
            self.si_yMax = self.fileHandle['HEADER']['Y_MAX'][0]
            self.si_xMax = self.fileHandle['HEADER']['X_MAX'][0]
            self.si_yMin = self.fileHandle['HEADER']['Y_MIN'][0]
            self.wkt = self.fileHandle['HEADER']['SPATIAL_REFERENCE'][0].decode()
        else:
            # set on setPixelGrid
            self.si_cnt = None
            self.si_idx = None
            self.si_binSize = None
            self.si_xMin = None
            self.si_yMax = None
            self.si_xMax = None
            self.si_yMin = None
            self.wkt = None
            
        # so we can be clever about when to read from disk
        self.lastExtent = None
        self.lastPoints = None
        self.lastPulses = None
        self.extent = None

    def getPixelGrid(self):
        pixGrid = pixelgrid.PixelGridDefn(projection=self.wkt,
                    xMin=self.si_xMin, xMax=self.si_xMax,
                    yMin=self.si_yMin, yMax=self.si_yMax,
                    xRes=self.si_binSize, yRes=self.si_binSize)
        return pixGrid
    
    def setPixelGrid(self, pixGrid):
        assert self.mode != generic.READ
        self.si_binSize = pixGrid.xRes
        self.si_xMin = pixGrid.xMin
        self.si_yMax = pixGrid.yMax
        self.si_xMax = pixGrid.xMax
        self.si_yMin = pixGrid.yMin
        self.wkt = pixGrid.projection
        
        # create spatial index
        (nrows, ncols) = pixGrid.getDimensions()
        self.si_cnt = numpy.zeros((ncols, nrows), dtype=numpy.int)
        self.si_idx = numpy.zeros((ncols, nrows), dtype=numpy.int)
    
    def setExtent(self, extent):
        self.extent = extent    
    
    def readPointsForExtent(self):
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
                        not self.lastPoints is None):
            return self.lastPoints
            
        # this should also return anything cached
        pulses = self.readPulsesForExtent()
        
        nReturns = pulses['NUMBER_OF_RETURNS']
        startIdxs = pulses['PTS_START_IDX']
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        pulse_bool = numpy.zeros(self.fileHandle['DATA']['POINTS'].shape,
                            numpy.bool)
    
        if len(startIdxs) > 0:    
            convertIdxBool(startIdxs, nReturns, pulse_bool)
        points = self.fileHandle['DATA']['POINTS'][pulse_bool]
        
        self.lastExtent = copy.copy(self.extent)
        self.lastPoints = points
        return points
            
    def readPulsesForExtent(self):
        """
        """
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
                        not self.lastPulses is None):
            return self.lastPulses
        
        tlxbin = int((self.extent.xMin - self.si_xMin) / self.si_binSize)
        tlybin = int((self.extent.yMax - self.si_yMax) / self.si_binSize)
        brxbin = int(numpy.ceil((self.extent.xMax - self.si_xMin) / self.si_binSize))
        brybin = int(numpy.ceil((self.si_yMax - self.extent.yMin) / self.si_binSize))
        
        # adjust for overlap
        tlxbin -= self.controls.overlap
        tlybin += self.controls.overlap
        brxbin += self.controls.overlap
        brybin -= self.controls.overlap
        
        if tlxbin < 0:
            tlxbin = 0
        if tlybin < 0:
            tlybin = 0
        if brxbin > self.si_cnt.shape[0]:
            brxbin = self.si_cnt.shape[0]
        if brybin > self.si_cnt.shape[1]:
            brybin = self.si_cnt.shape[1]
        
        cnt_subset = self.si_cnt[tlxbin:brxbin+1, tlybin:brybin+1].flatten()
        idx_subset = self.si_idx[tlxbin:brxbin+1, tlybin:brybin+1].flatten()
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        pulse_bool = numpy.zeros(self.fileHandle['DATA']['PULSES'].shape,
                                numpy.bool)
        if len(idx_subset) > 0:
            convertIdxBool(idx_subset, cnt_subset, pulse_bool)
        pulses = self.fileHandle['DATA']['PULSES'][pulse_bool]
                                
        self.lastExtent = copy.copy(self.extent)
        self.lastPulses = pulses
        self.lastPoints = None # are now invalid
        return pulses
    
    
    def writePointsForExtent(self, points):
        # TODO: must remove points in overlap area
        raise NotImplementedError()
    def writePulsesForExtent(self, pulses):
        # TODO: must remove points in overlap area
        raise NotImplementedError()
    # see below for no spatial index
    def readPoints(self, n):
        raise NotImplementedError()
    def readPulses(self, n):
        raise NotImplementedError()
    def writePoints(self, points):
        raise NotImplementedError()
    def writePulses(self, pulses):
        raise NotImplementedError()

    def close(self):
        if self.mode != generic.READ:
            # write out to file
            self.fileHandle['INDEX']['PLS_PER_BIN'] = self.si_cnt
            self.fileHandle['INDEX']['BIN_OFFSETS'] = self.si_idx
            self.fileHandle['HEADER']['BIN_SIZE'] = self.si_binSize
            self.fileHandle['HEADER']['X_MIN'] = self.si_xmin
            self.fileHandle['HEADER']['Y_MAX'] = self.si_ymax
            self.fileHandle['HEADER']['X_MAX'] = self.si_xmax
            self.fileHandle['HEADER']['Y_MIN'] = self.si_ymin
            self.fileHandle['HEADER']['SPATIAL_REFERENCE'] = self.wkt
            
        # close
        self.fileHandle.close()
        self.fileHandle = None        
        self.lastExtent = None
        self.lastPoints = None
        self.lastPulses = None
