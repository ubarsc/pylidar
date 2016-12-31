"""
Functions for voxelization of TLS scans (Hancock et al., 2016)
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

import numpy
from numba import jit
from osgeo import gdal

from pylidar import lidarprocessor


def run_voxel_hancock2016(dataFiles, controls, otherargs, outfiles):
    """
    Main function for VOXEL_HANCOCK2016
    
    The gap fraction of each voxel is the ratio of the number of beams that reach the voxel 
    to the number that could have passed through.
    
    """    
    # define 3D voxel grid
    otherargs.nX = int( (otherargs.bounds[3] - otherargs.bounds[0]) / otherargs.voxelsize[0] )
    otherargs.nY = int( (otherargs.bounds[4] - otherargs.bounds[1]) / otherargs.voxelsize[1] )
    otherargs.nZ = int( (otherargs.bounds[5] - otherargs.bounds[2]) / otherargs.voxelsize[2] )
    otherargs.voxDimX = otherargs.bounds[3] - otherargs.bounds[0]
    otherargs.voxDimY = otherargs.bounds[4] - otherargs.bounds[1]
    otherargs.voxDimZ = otherargs.bounds[5] - otherargs.bounds[2]    
    
    # initialize voxel arrays    
    otherargs.outgrids = dict()
    nVox = otherargs.nX * otherargs.nY * otherargs.nZ
    otherargs.outgrids["hits"] = numpy.zeros(nVox, dtype=numpy.uint32)
    otherargs.outgrids["miss"] = numpy.zeros(nVox, dtype=numpy.uint32)
    otherargs.outgrids["wcnt"] = numpy.zeros(nVox, dtype=numpy.float32)
    otherargs.outgrids["scan"] = numpy.zeros(nVox, dtype=numpy.uint8)
    
    # run the voxelization
    nScans = len(dataFiles.inList)
    for i in range(nScans):
        otherargs.scan = i
        print("Processing %s" % dataFiles.inList[i].fname)        
        temphits = numpy.copy(otherargs.outgrids["hits"])
        lidarprocessor.doProcessing(runVoxelization, dataFiles, controls=controls, otherArgs=otherargs)
        otherargs.outgrids["scan"] += numpy.uint8(otherargs.outgrids["hits"] > temphits)
    
    # write output to an image file
    for gridname, outfile in zip(otherargs.outgrids.keys(), outfiles):
        otherargs.outgrids[gridname].shape = (otherargs.nZ, otherargs.nY, otherargs.nX)
        saveVoxels(outfile, otherargs.outgrids[gridname], otherargs.bounds[0], otherargs.bounds[1], otherargs.voxelsize, proj=otherargs.proj, drivername=otherargs.rasterdriver)
        

def runVoxelization(data, otherargs):
    """
    Voxelization function for the lidar processor
    """
    # read the pulse data
    pulsecolnames = ['NUMBER_OF_RETURNS','ZENITH','AZIMUTH','X_ORIGIN','Y_ORIGIN','Z_ORIGIN']       
    pulses = data.inList[otherargs.scan].getPulses(colNames=pulsecolnames)   
    
    if pulses.shape[0] > 0:
    
        # read the point data
        pointcolnames = ['X','Y','Z','RETURN_NUMBER']
        pointsByPulses = data.inList[otherargs.scan].getPointsByPulse(colNames=pointcolnames)

        # unit direction vector
        theta = numpy.radians(pulses['ZENITH'])
        phi = numpy.radians(pulses['AZIMUTH'])
        dx = numpy.sin(theta) * numpy.sin(phi)
        dy = numpy.sin(theta) * numpy.cos(phi)
        dz = numpy.cos(theta)

        # temporary arrays
        max_nreturns = numpy.max(pulses['NUMBER_OF_RETURNS'])
        voxIdx = numpy.empty(max(1,max_nreturns), dtype=numpy.uint32)

        # run the voxelization
        runTraverseVoxels(pulses['X_ORIGIN'], pulses['Y_ORIGIN'], pulses['Z_ORIGIN'], \
            pointsByPulses['X'].data, pointsByPulses['Y'].data, pointsByPulses['Z'].data, dx, dy, dz, \
            pulses['NUMBER_OF_RETURNS'], otherargs.voxDimX, otherargs.voxDimY, otherargs.voxDimZ, \
            otherargs.nX, otherargs.nY, otherargs.nZ, otherargs.bounds, otherargs.voxelsize, \
            otherargs.outgrids["hits"], otherargs.outgrids["miss"], otherargs.outgrids["wcnt"], voxIdx)


@jit(nopython=True)
def runTraverseVoxels(x0, y0, z0, x1, y1, z1, dx, dy, dz, number_of_returns, voxDimX, voxDimY, voxDimZ, \
                      nX, nY, nZ, bounds, voxelSize, hitsArr, missArr, wcntArr, voxIdx):
    """
    Loop through each pulse and run voxel traversal
    """
    for i in range(number_of_returns.shape[0]):
        
        traverseVoxels(x0[i], y0[i], z0[i], x1[:,i], y1[:,i], z1[:,i], dx[i], dy[i], dz[i], \
            nX, nY, nZ, voxDimX, voxDimY, voxDimZ, bounds, voxelSize, number_of_returns[i], \
            hitsArr, missArr, wcntArr, voxIdx)
    

@jit(nopython=True)
def traverseVoxels(x0, y0, z0, x1, y1, z1, dx, dy, dz, nX, nY, nZ, voxDimX, voxDimY, voxDimZ, \
               bounds, voxelSize, number_of_returns, hitsArr, missArr, wcntArr, voxIdx):
    """
    A fast and simple voxel traversal algorithm through a 3D voxel space (J. Amanatides and A. Woo, 1987)
    Inputs:
       x0, y0, z0
       x1, y1, z1
       dx, dy, dz
       nX, nY, nZ
       bounds
       voxelSize
       number_of_returns
    Outputs:
       hitsArr
       missArr
       wcntArr
    """
    intersect, tmin = gridIntersection(x0, y0, z0, dx, dy, dz, bounds)    
    if intersect == 1:
        
        if tmin < 0:
            tmin = 0.0

        startX = x0 + tmin * dx
        startY = y0 + tmin * dy
        startZ = z0 + tmin * dz
        
        x = numpy.floor( ((startX - bounds[0]) / voxDimX) * nX )
        y = numpy.floor( ((startY - bounds[1]) / voxDimY) * nY )
        z = numpy.floor( ((startZ - bounds[2]) / voxDimZ) * nZ )               
        
        for i in range(number_of_returns):
            px = numpy.floor( ((x1[i] - bounds[0]) / voxDimX) * nX )
            py = numpy.floor( ((y1[i] - bounds[1]) / voxDimY) * nY )
            pz = numpy.floor( ((z1[i] - bounds[2]) / voxDimZ) * nZ )
            voxIdx[i] = int(px + nX * py + nX * nY * pz)   
        
        if x == nX:
            x -= 1
        if y == nY:
            y -= 1           
        if z == nZ:
            z -= 1
         
        if dx >= 0:
            tVoxelX = (x + 1) / nX
            stepX = 1
        else:
            tVoxelX = x / nX
            stepX = -1
        
        if dy >= 0:
            tVoxelY = (y + 1) / nY
            stepY = 1
        else:
            tVoxelY = y / nY
            stepY = -1
        
        if dz >= 0:
            tVoxelZ = (z + 1) / nZ
            stepZ = 1
        else:
            tVoxelZ = z / nZ
            stepZ = -1
                
        voxelMaxX = bounds[0] + tVoxelX * voxDimX
        voxelMaxY = bounds[1] + tVoxelY * voxDimY
        voxelMaxZ = bounds[2] + tVoxelZ * voxDimZ

        if dx == 0:
            tMaxX = 1.0
            tDeltaX = 1.0
        else:
            tMaxX = tmin + (voxelMaxX - startX) / dx
            tDeltaX = voxelSize[0] / abs(dx)
            
        if dy == 0:    
            tMaxY = 1.0
            tDeltaY = 1.0
        else:
            tMaxY = tmin + (voxelMaxY - startY) / dy
            tDeltaY = voxelSize[1] / abs(dy)
            
        if dz == 0:
            tMaxZ = 1.0
            tDeltaZ = 1.0
        else:
            tMaxZ = tmin + (voxelMaxZ - startZ) / dz
            tDeltaZ = voxelSize[2] / abs(dz) 
                
        foundLast = 0  
        while (x < nX) and (x >= 0) and (y < nY) and (y >= 0) and (z < nZ) and (z >= 0):
                        
            vidx = int(x + nX * y + nX * nY * z)
            if foundLast == 0:
                hitsArr[vidx] += 1
            else:
                missArr[vidx] += 1
            
            for i in range(number_of_returns):                
                if (vidx == voxIdx[i]):
                    wcntArr[vidx] += 1.0 / number_of_returns
                    if i == number_of_returns-1:                    
                        foundLast = 1
            
            if tMaxX < tMaxY:
                if tMaxX < tMaxZ:
                    x += stepX
                    tMaxX += tDeltaX
                else:
                    z += stepZ
                    tMaxZ += tDeltaZ
            else:
                if tMaxY < tMaxZ:
                    y += stepY
                    tMaxY += tDeltaY           
                else:
                    z += stepZ
                    tMaxZ += tDeltaZ


@jit(nopython=True)
def gridIntersection(x0, y0, z0, dx, dy, dz, bounds):
    """
    Voxel grid intersection test using Smits algorithm
    Inputs:
       x0, y0, z0
       dz, dy, dz
       bounds
    Outputs:
       intersect: 0 = no intersection, 1 = intersection
       tmin: distance from the beam origin
    """
    if (dx > 0) or (dx < 0):
        divX = 1.0 / dx
    else:
        divX = 1.0
    
    if divX > 0:
    	tmin = (bounds[0] - x0) * divX
    	tmax = (bounds[3] - x0) * divX
    else:
    	tmin = (bounds[3] - x0) * divX
    	tmax = (bounds[0] - x0) * divX
      
    if (dy > 0) or (dy < 0):
        divY = 1.0 / dy
    else:
        divY = 1.0
    
    if divY >= 0:
        tymin = (bounds[1] - y0) * divY
        tymax = (bounds[4] - y0) * divY
    else:
    	tymin = (bounds[4] - y0) * divY
    	tymax = (bounds[1] - y0) * divY
    
    if (tmin > tymax) or (tymin > tmax):
        intersect = 0
        tmin = -1.0
    else:
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        if (dz > 0) or (dz < 0):
            divZ = 1.0 / dz
        else:
            divZ = 1.0
        
        if divZ >= 0:
            tzmin = (bounds[2] - z0) * divZ
            tzmax = (bounds[5] - z0) * divZ
        else:
            tzmin = (bounds[5] - z0) * divZ
            tzmax = (bounds[2] - z0) * divZ

        if (tmin > tzmax) or (tzmin > tmax):
            intersect = 0
            tmin = -1.0
        else:
            if tzmin > tmin:
                tmin = tzmin
            if tzmax < tmax:
                tmax = tzmax
            intersect = 1
    
    return intersect,tmin
    

def saveVoxels(outfile, vox, xmin, ymax, res, nullval=None, proj=None, drivername='HFA'):
    """
    Save the given 3D voxel array as a multiband GDAL raster file. 
    """
    if vox.ndim == 2:
        (nRows, nCols) = vox.shape
        vox.shape = (1, nRows, nCols)
    (nBins, nRows, nCols) = vox.shape
    
    gdalTypeDict = {numpy.dtype(numpy.float32):gdal.GDT_Float32,
                    numpy.dtype(numpy.uint16):gdal.GDT_UInt16,
                    numpy.dtype(numpy.uint32):gdal.GDT_UInt32,
                    numpy.dtype(numpy.uint8):gdal.GDT_Byte}
    gdalType = gdalTypeDict[vox.dtype]
    
    if vox.dtype == 'float32':
        nullval = 0.0
    else:
        nullval = numpy.iinfo(vox.dtype).max
    
    drvr = gdal.GetDriverByName(drivername)
    ds = drvr.Create(outfile, nCols, nRows, nBins, gdalType, ['COMPRESS=YES'])
    for i in range(nBins):
        band = ds.GetRasterBand(i+1)
        band.WriteArray(vox[i,:,:])
        band.SetNoDataValue(nullval)
    
    if proj is not None:
        ds.SetProjection(proj)
    
    ds.SetGeoTransform((xmin, res[0], 0, ymax, 0, -res[1]))
