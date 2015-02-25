
"""
SPD V3
"""
import copy
import numpy
import h5py
from numba import jit
from rios import pixelgrid
from . import generic

# so we can check the user has passed in expected array type
PULSE_DTYPE = numpy.dtype([('GPS_TIME', '<u8'), ('PULSE_ID', '<u8'), 
('X_ORIGIN', '<f8'), ('Y_ORIGIN', '<f8'), ('Z_ORIGIN', '<f4'), 
('H_ORIGIN', '<f4'), ('X_IDX', '<f8'), ('Y_IDX', '<f8'), ('AZIMUTH', '<f4'), 
('ZENITH', '<f4'), ('NUMBER_OF_RETURNS', 'u1'), 
('NUMBER_OF_WAVEFORM_TRANSMITTED_BINS', '<u2'), 
('NUMBER_OF_WAVEFORM_RECEIVED_BINS', '<u2'), ('RANGE_TO_WAVEFORM_START', '<f4'),
('AMPLITUDE_PULSE', '<f4'), ('WIDTH_PULSE', '<f4'), ('USER_FIELD', '<u4'), 
('SOURCE_ID', '<u2'), ('SCANLINE', '<u4'), ('SCANLINE_IDX', '<u2'), 
('RECEIVE_WAVE_NOISE_THRES', '<f4'), ('TRANS_WAVE_NOISE_THRES', '<f4'), 
('WAVELENGTH', '<f4'), ('RECEIVE_WAVE_GAIN', '<f4'), 
('RECEIVE_WAVE_OFFSET', '<f4'), ('TRANS_WAVE_GAIN', '<f4'), 
('TRANS_WAVE_OFFSET', '<f4'), ('PTS_START_IDX', '<u8'), 
('TRANSMITTED_START_IDX', '<u8'), ('RECEIVED_START_IDX', '<u8')])

POINT_DTYPE = numpy.dtype([('RETURN_ID', 'u1'), ('GPS_TIME', '<f8'), 
('X', '<f8'), ('Y', '<f8'), ('Z', '<f4'), ('HEIGHT', '<f4'), ('RANGE', '<f4'), 
('AMPLITUDE_RETURN', '<f4'), ('WIDTH_RETURN', '<f4'), ('RED', '<u2'), 
('GREEN', '<u2'), ('BLUE', '<u2'), ('CLASSIFICATION', 'u1'), 
('USER_FIELD', '<u4'), ('IGNORE', 'u1'), ('WAVE_PACKET_DESC_IDX', '<i2'), 
('WAVEFORM_OFFSET', '<u4')])

    
@jit
def BuildSpatialIndex(binNum, sortedBinNumNdx, si_start, si_count):
    nCols = si_start.shape[1]
    nThings = binNum.shape[0]
    for i in range(nThings):
        bn = binNum[sortedBinNumNdx[i]]
        row = bn // nCols
        col = bn % nCols
        if si_count[row, col] == 0:
            si_start[row, col] = i
        si_count[row, col] += 1
    
@jit
def convertIdxBool(start_idx_array, count_array, outBool, outRow, outCol, outIdx, counts, outMask):
    """
    Convert SPD's default regular spatial index of pulse offsets and pulse counts
    per bin into a single boolean array. It is assumed that out is already
    set to all False

    Written to use numba - numpy version was very slow. Anyone any ideas
    on how to do this quickly in normal numpy?
    """
    # start_idx_array - array of start indexes
    # count_array - array of counts
    # outBool - same shape as the dataset size, but bool inited to False
    # outIdx 3d - (max(count_array), nRows, nCols) int32 inited to 0
    # outMask 3d - bool same shape as outIdx inited to True
    # outRow same shape as outBool but int64 cerate with numpy.empty()
    # outCol same shape as outBool but int64 empty()
    # counts (nRows, nCols) int32 inited to 0
    
    nRows = start_idx_array.shape[0]
    nCols = start_idx_array.shape[1]
    
    for col in range(nCols):
        for row in range(nRows):
        
            cnt = count_array[row, col]
            startidx = start_idx_array[row, col]
            for i in range(cnt):
                # seems a strange bug in numba/llvm where the
                # result of this add gets promoted to a double
                # so cast it back
                idx = int(startidx + i)
                outBool[idx] = True
                outRow[idx] = row
                outCol[idx] = col
                
    n = outBool.shape[0]
    counter = 0
    for i in range(n):
        if outBool[i]:
            row = outRow[i]
            col = outCol[i]
            c = counts[row, col]
            outIdx[c, row, col] = counter
            outMask[c, row, col] = False
            counts[row, col] += 1
            counter += 1
                
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
            # TODO: handle when the spatial index does not exist
            self.si_cnt = self.fileHandle['INDEX']['PLS_PER_BIN'][...]
            self.si_idx = self.fileHandle['INDEX']['BIN_OFFSETS'][...]
            self.si_binSize = self.fileHandle['HEADER']['BIN_SIZE'][0]
            self.si_xMin = self.fileHandle['HEADER']['X_MIN'][0]
            self.si_yMax = self.fileHandle['HEADER']['Y_MAX'][0]
            # bottom right coords don't seem right (of data rather than si)
            self.si_xMax = self.si_xMin + (self.si_idx.shape[1] * self.si_binSize)
            self.si_yMin = self.si_yMax - (self.si_idx.shape[0] * self.si_binSize)
            
            #self.si_xMax = self.fileHandle['HEADER']['X_MAX'][0]
            #self.si_yMin = self.fileHandle['HEADER']['Y_MIN'][0]
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
        point_bool, point_idx, point_idx_mask = self.convertIdxToUsefulStuff(startIdxs,
                    nReturns, self.fileHandle['DATA']['POINTS'].shape[0])
        
        points = self.fileHandle['DATA']['POINTS'][point_bool]
        
        self.lastExtent = copy.copy(self.extent)
        self.lastPoints = points
        # TODO: set to None in constructor
        self.lastPoints_Idx = points_idx
        self.lastPoints_IdxMask = points_idx_mask
        return points
            
    def readPulsesForExtent(self):
        """
        """
        # returned cached if possible
        if (self.lastExtent is not None and self.lastExtent == self.extent and 
                        not self.lastPulses is None):
            return self.lastPulses

        tlxbin = int((self.extent.xMin - self.si_xMin) / self.si_binSize)
        tlybin = int((self.si_yMax - self.extent.yMax) / self.si_binSize)
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
        if brxbin > self.si_cnt.shape[1]:
            brxbin = self.si_cnt.shape[1]
        if brybin > self.si_cnt.shape[0]:
            brybin = self.si_cnt.shape[0]
        
        cnt_subset = self.si_cnt[tlybin:brybin, tlxbin:brxbin]
        idx_subset = self.si_idx[tlybin:brybin, tlxbin:brxbin]
        
        # h5py prefers to take it's index by numpy bool array
        # of the same shape as the dataset
        # so we do this. If you give it the indices themselves
        # this must be done as a list which is slow
        pulse_bool, pulse_idx, pulse_idx_mask = self.convertIdxToUsefulStuff(idx_subset, cnt_subset, 
                self.fileHandle['DATA']['PULSES'].shape[0])
        pulses = self.fileHandle['DATA']['PULSES'][pulse_bool]
        
        self.lastExtent = copy.copy(self.extent)
        self.lastPulses = pulses
        self.lastPulses_Idx = pulse_idx
        self.lastPulses_IdxMask = pulse_idx_mask
        self.lastPoints = None # are now invalid
        return pulses

    @staticmethod
    def convertIdxToUsefulStuff(start_idx_array, count_array, outSize):
        outBool = numpy.zeros((outSize,), dtype=numpy.bool)
        maxCount = count_array.max()
        nRows, nCols = count_array.shape
        outIdx = numpy.zeros((maxCount, nRows, nCols), dtype=numpy.int32)
        outMask = numpy.ones((maxCount, nRows, nCols), numpy.bool)
        outRow = numpy.empty((outSize,), dtype=numpy.int64)
        outCol = numpy.empty((outSize,), dtype=numpy.int64)
        counts = numpy.zeros((nRows, nCols), dtype=numpy.int32)
        
        convertIdxBool(start_idx_array, count_array, outBool, outRow, outCol, outIdx, counts, outMask)
        
        return outBool, outIdx, outMask

        # TODO: a method to create a ragged array of points-by-pulses using PTS_START_IDX 
        
        # TODO: a method to create 2-d ragged array of points by pixel
        # TODO: a method to create 2-d ragged array of pulses by pixel
    
    def readTransmitted(self, pulse):
        # TODO: update so an array of pulses is taken
        # and a masked array is returned
        idx = pulse['TRANSMITTED_START_IDX']
        cnt = pulse['NUMBER_OF_WAVEFORM_TRANSMITTED_BINS']
        transmitted = self.fileHandle['DATA']['TRANSMITTED'][idx:idx+cnt+1]
        return transmitted
        
    def readReceived(self, pulse):
        # TODO: update so an array of pulses is taken
        # and a masked array is returned
        idx = pulse['RECEIVED_START_IDX']
        cnt = pulse['NUMBER_OF_WAVEFORM_RECEIVED_BINS']
        received = self.fileHandle['DATA']['RECEIVED'][idx:idx+cnt+1]
        return received
    
    def writePointsForExtent(self, points):
        # TODO: must remove points in overlap area
        # somehow? Via Pulses?
        assert self.mode != generic.READ
        raise NotImplementedError()
        
    # TODO: write both at once 
    def writePulsesForExtent(self, pulses):
        assert self.mode != generic.READ
        # we are fussy here about the dtype - the format
        # written must match the spec. Not such an issue for SPD v4?
        if pulses.dtype != PULSE_DTYPE:
            msg = ("Invalid pulse array. " +
                "Fields and types must be the same as that read")
            raise LiDARInvalidData(msg)
        
        # self.extent is the size of the block without the overlap
        # so just strip out everything outside of it
        mask = ( (pulses['X_IDX'] >= self.extent.xMin) & 
                    (pulses['X_IDX'] <= self.extent.xMax) & 
                    (pulses['Y_IDX'] >= self.extent.yMin) &
                    (pulses['Y_IDX'] <= self.extent.yMax))
        pulses = pulses[mask]
        
        # TOOD: Points must be written at the same time so 
        # we can set PTS_START_IDX
        
        if self.mode == generic.CREATE:
            # need to extend the hdf5 dataset before writing
            oldSize = self.fileHandle['DATA']['PULSES'].shape[0]
            nPulses = len(pulses)
            newSize = oldSize + nPulses
            self.fileHandle['DATA']['PULSES'].resize((newSize,))
            
            
        else:
            # mode == WRITE
            # TODO: not totally sure what this means at the moment
            pass
        
        # now update the spatial index
        raise NotImplementedError()

    @staticmethod
    def NeilsSpatialWrapper(coordOne, coordTwo, binSize, coordOneMin, coordTwoMin, nRows, nCols):
        # coordOne is the coordinate corresponding to bin row. CoordTwo corresponds to bin col.
        # Note that coordOne will always be reversed??????
        sortedIdx = numpy.empty_like(coordOne, dtype=numpy.int64)
        row = numpy.floor((coordOneMax - coordOne) / binSize).astype(numpy.uint32)
        col = numpy.floor((coordTwo - coordTwoMin) / binSize).astype(numpy.uint32)
        binNum = col * nCols + row
        sortedBinNumNdx = numpy.argsort(binNum)
    
        si_start = numpy.zeros((nRows, nCols), dtype=numpy.uint32)
        si_count = numpy.zeros((nRows, nCols), dtype=numpy.uint32)
    
        BuildSpatialIndex(binNum, sortedBinNumNdx, si_start, si_count)
        return sortedBinNumNdx, si_start, si_count
        
    def writeTransmitted(self, pulse, transmitted):
        raise NotImplementedError()
        
    def writeReceived(self, pulse, received):
        raise NotImplementedError()

    def hasSpatialIndex(self):
        # assume this format always does
        # TODO: is this correct?
        return True
        
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
