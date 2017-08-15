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

import os
import numpy
import collections
from numba import jit

from pylidar.toolbox import spatial
from pylidar.toolbox.canopy import canopycommon
from pylidar import lidarprocessor

VOXEL_SCALE = 10000
VOXEL_OFFSET = 0
VOXEL_NULL = 9999


def classify_voxels(hits, miss, occl, classification, ground=None):
    """
    Classification of voxels
    
    Class       Value   Hits    Misses   Occluded
    Observed    5       >0      >=0      >=0
    Empty       4       =0      >0       >=0
    Hidden      3       =0      =0       >0
    Unobserved  2       =0      =0       =0
    Ground      1
    """
    class_tmp = numpy.zeros_like(classification)
    
    mask = (hits > 0) & (miss >= 0) & (occl >= 0)
    class_tmp[mask] = 5
    
    mask = (hits == 0) & (miss > 0) & (occl >= 0)
    class_tmp[mask] = 4
    
    mask = (hits == 0) & (miss == 0) & (occl > 0)
    class_tmp[mask] = 3
    
    mask = (hits == 0) & (miss == 0) & (occl == 0)
    class_tmp[mask] = 2
    
    classification = numpy.maximum(class_tmp, classification)
    
    if ground is not None:
        classification[ground == 1] = 1
    
    return classification 


def run_voxel_hancock2016(infiles, controls, otherargs, outfiles):
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
    
    # initialize summary voxel arrays
    summaryOutputs = ["scans","cover","class"]   
    otherargs.outgrids = collections.OrderedDict()
    nVox = otherargs.nZ * otherargs.nY * otherargs.nX
    otherargs.outgrids["scans"] = numpy.zeros(nVox, dtype=numpy.uint8)
    otherargs.outgrids["cover"] = numpy.zeros(nVox, dtype=numpy.float32)
    otherargs.outgrids["class"] = numpy.zeros(nVox, dtype=numpy.uint8)
    wtot = numpy.zeros(nVox, dtype=numpy.float32)
        
    # set ground boundary for voxel traversal
    if otherargs.externaldem is not None:                    
        if (otherargs.dataDem.shape[1] != otherargs.nX) or (otherargs.dataDem.shape[0] != otherargs.nY):
            msg = 'External DEM size X and Y dimensions must be the same as the voxel grid'
            raise spatial.SpatialException(msg)            
        otherargs.outgrids["gvox"] = numpy.zeros((otherargs.nZ,otherargs.nY,otherargs.nX), dtype=numpy.uint8)
        x,y = numpy.meshgrid(numpy.arange(otherargs.nX), numpy.arange(otherargs.nY))            
        z = ((otherargs.dataDem - otherargs.bounds[2]) / otherargs.voxelsize[2]).astype(numpy.uint)            
        otherargs.outgrids["gvox"][z,y,x] = 1
        otherargs.outgrids["gvox"] = otherargs.outgrids["gvox"].reshape(nVox)
    else:
        otherargs.outgrids["gvox"] = numpy.zeros(nVox, dtype=numpy.uint8)    
    
    # loop through each scan
    outputSuffix = os.path.splitext(outfiles[0])[1]
    scanOutputs = ["hits","miss","occl","pgap"]
    for i,infile in enumerate(infiles):
        
        # initialize scan voxel arrays 
        otherargs.scangrids = collections.OrderedDict()     
        otherargs.scangrids["hits"] = numpy.zeros(nVox, dtype=numpy.float32)
        otherargs.scangrids["miss"] = numpy.zeros(nVox, dtype=numpy.float32)
        otherargs.scangrids["occl"] = numpy.zeros(nVox, dtype=numpy.float32)
        otherargs.scangrids["plen"] = numpy.ones(nVox, dtype=numpy.float32)
        
        # run the voxelization                
        print("Voxel traversing %s" % infile)
        dataFiles = canopycommon.prepareInputFiles(infiles, otherargs, index=i)       
        lidarprocessor.doProcessing(runVoxelization, dataFiles, controls=controls, otherArgs=otherargs)
        
        # calculate scan grids
        nshots = otherargs.scangrids["miss"] + otherargs.scangrids["hits"]
        otherargs.scangrids["pgap"] = numpy.where(nshots > 0, otherargs.scangrids["miss"] / nshots, VOXEL_NULL)
        w = numpy.where(nshots > 0, (nshots / (nshots + otherargs.scangrids["occl"])) * otherargs.scangrids["plen"], 0)
        
        # update summary grids
        otherargs.outgrids["scans"] += numpy.uint8(nshots > 0)
        otherargs.outgrids["cover"] += numpy.where(nshots > 0, (1 - otherargs.scangrids["pgap"]) * w, 0)
        wtot += w
        otherargs.outgrids["class"] = classify_voxels(otherargs.scangrids["hits"], otherargs.scangrids["miss"], \
            otherargs.scangrids["occl"], otherargs.outgrids["class"], ground=otherargs.outgrids["gvox"])
        
        # write output scan voxel arrays to image files
        for gridname in scanOutputs:
            outfile = "%s.%s%s" % (os.path.splitext(infile)[0], gridname, outputSuffix)
            iw = spatial.ImageWriter(outfile, tlx=otherargs.bounds[0], tly=otherargs.bounds[4], binSize=otherargs.voxelsize[0], \
                 driverName=otherargs.rasterdriver, epsg=otherargs.proj[0], numBands=otherargs.nZ, nullVal=VOXEL_NULL)
            otherargs.scangrids[gridname].shape = (otherargs.nZ, otherargs.nY, otherargs.nX)
            for i in range(otherargs.nZ):
                iw.setLayer(otherargs.scangrids[gridname][i], layerNum=i+1)
            iw.close()
    
    # calculate vertical cover profiles using conditional probability
    otherargs.outgrids["cover"] = numpy.where(wtot > 0, otherargs.outgrids["cover"] / wtot, 0)
    otherargs.outgrids["cover"].shape = (otherargs.nZ, otherargs.nY, otherargs.nX)
    otherargs.outgrids["gvox"].shape = (otherargs.nZ, otherargs.nY, otherargs.nX)
    otherargs.outgrids["gvox"] = numpy.cumsum(otherargs.outgrids["gvox"][::-1], axis=0)[::-1]
    n = otherargs.nZ - 1
    for i in range(n-1,-1,-1):
        p_o = otherargs.outgrids["cover"][i+1]
        p_i = otherargs.outgrids["cover"][i]
        otherargs.outgrids["cover"][i] = p_o + (1 - p_o) * p_i
        otherargs.outgrids["cover"][i] = numpy.where(otherargs.outgrids["gvox"][i] == 0, \
            otherargs.outgrids["cover"][i], VOXEL_NULL)
    
    # write output summary voxel arrays to image files
    otherargs.outgrids["scans"].shape = (otherargs.nZ, otherargs.nY, otherargs.nX)
    otherargs.outgrids["class"].shape = (otherargs.nZ, otherargs.nY, otherargs.nX)
    for i,gridname in enumerate(summaryOutputs):
        iw = spatial.ImageWriter(outfiles[i], tlx=otherargs.bounds[0], tly=otherargs.bounds[4], binSize=otherargs.voxelsize[0], \
             driverName=otherargs.rasterdriver, epsg=otherargs.proj[0], numBands=otherargs.nZ, nullVal=VOXEL_NULL)
        for j in range(otherargs.nZ):
            iw.setLayer(otherargs.outgrids[gridname][j], layerNum=j+1)
        iw.close()
   

def runVoxelization(data, otherargs):
    """
    Voxelization function for the lidar processor
    """
    # read the pulse data
    pulsecolnames = ['NUMBER_OF_RETURNS','ZENITH','AZIMUTH','X_ORIGIN','Y_ORIGIN','Z_ORIGIN']       
    pulses = data.inFiles[0].getPulses(colNames=pulsecolnames)   
    
    if pulses.shape[0] > 0:
        
        # read the point data
        if otherargs.lidardriver[0] == "SPDV3":
            pointcolnames = ['X','Y','Z','RANGE','CLASSIFICATION','RETURN_ID']
        else:
            pointcolnames = ['X','Y','Z','RANGE','CLASSIFICATION','RETURN_NUMBER']            
            pulses['ZENITH'] = numpy.radians(pulses['ZENITH'])
            pulses['AZIMUTH'] = numpy.radians(pulses['AZIMUTH'])
        pointsByPulses = data.inFiles[0].getPointsByPulse(colNames=pointcolnames)
        
        # calculate the unit direction vector
        dx = numpy.sin(pulses['ZENITH']) * numpy.sin(pulses['AZIMUTH'])
        dy = numpy.sin(pulses['ZENITH']) * numpy.cos(pulses['AZIMUTH'])
        dz = numpy.cos(pulses['ZENITH'])
        
        # temporary arrays
        max_nreturns = numpy.max(pulses['NUMBER_OF_RETURNS'])
        voxIdx = numpy.empty(max_nreturns, dtype=numpy.uint32)
        
        # run the voxelization
        runTraverseVoxels(pulses['X_ORIGIN'], pulses['Y_ORIGIN'], pulses['Z_ORIGIN'], \
            pointsByPulses['X'].data, pointsByPulses['Y'].data, pointsByPulses['Z'].data, dx, dy, dz, \
            pulses['NUMBER_OF_RETURNS'], otherargs.voxDimX, otherargs.voxDimY, otherargs.voxDimZ, \
            otherargs.nX, otherargs.nY, otherargs.nZ, otherargs.bounds, otherargs.voxelsize, \
            otherargs.scangrids["hits"], otherargs.scangrids["miss"], otherargs.scangrids["occl"], \
            otherargs.outgrids["gvox"], otherargs.scangrids["plen"], voxIdx)


@jit(nopython=True)
def runTraverseVoxels(x0, y0, z0, x1, y1, z1, dx, dy, dz, number_of_returns, voxDimX, voxDimY, voxDimZ, \
                      nX, nY, nZ, bounds, voxelSize, hitsArr, missArr, occlArr, gvoxArr, plenArr, voxIdx):
    """
    Loop through each pulse and run voxel traversal
    """
    for i in range(number_of_returns.shape[0]):        
        traverseVoxels(x0[i], y0[i], z0[i], x1[:,i], y1[:,i], z1[:,i], dx[i], dy[i], dz[i], \
            nX, nY, nZ, voxDimX, voxDimY, voxDimZ, bounds, voxelSize, number_of_returns[i], \
            hitsArr, missArr, occlArr, gvoxArr, plenArr, voxIdx)
    

@jit(nopython=True)
def traverseVoxels(x0, y0, z0, x1, y1, z1, dx, dy, dz, nX, nY, nZ, voxDimX, voxDimY, voxDimZ, \
               bounds, voxelSize, number_of_returns, hitsArr, missArr, occlArr, gvoxArr, plenArr, voxIdx):
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
       gvoxArr
       voxIdx
    Outputs:
       hitsArr
       missArr
       occlArr
       plenArr
    """
    intersect, tmin, tmax = gridIntersection(x0, y0, z0, dx, dy, dz, bounds)    
    if intersect == 1:
        
        tmin = max(0, tmin)
        tmax = min(1, tmax)

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
         
        if dx > 0:
            tVoxelX = (x + 1) / nX
            stepX = 1
        elif dx < 0:
            tVoxelX = x / nX
            stepX = -1
        else:
            tVoxelX = (x + 1) / nX
            stepX = 0
        
        if dy > 0:
            tVoxelY = (y + 1) / nY
            stepY = 1
        elif dy < 0:
            tVoxelY = y / nY
            stepY = -1
        else:
            tVoxelY = (y + 1) / nY
            stepY = 0  
        
        if dz > 0:
            tVoxelZ = (z + 1) / nZ
            stepZ = 1
        elif dz < 0:
            tVoxelZ = z / nZ
            stepZ = -1
        else:
            tVoxelZ = (z + 1) / nZ
            stepZ = 0            
        
        voxelMaxX = bounds[0] + tVoxelX * voxDimX
        voxelMaxY = bounds[1] + tVoxelY * voxDimY
        voxelMaxZ = bounds[2] + tVoxelZ * voxDimZ
        
        if dx == 0:
            tMaxX = tmax
            tDeltaX = tmax
        else:
            tMaxX = tmin + (voxelMaxX - startX) / dx
            tDeltaX = voxelSize[0] / abs(dx)
            
        if dy == 0:    
            tMaxY = tmax
            tDeltaY = tmax
        else:
            tMaxY = tmin + (voxelMaxY - startY) / dy
            tDeltaY = voxelSize[1] / abs(dy)
            
        if dz == 0:
            tMaxZ = tmax
            tDeltaZ = tmax
        else:
            tMaxZ = tmin + (voxelMaxZ - startZ) / dz
            tDeltaZ = voxelSize[2] / abs(dz)
        
        gnd = 0
        wmiss = 1.0
        woccl = 0.0
        if number_of_returns > 0:
            w = 1.0 / number_of_returns
        else:
            w = 0.0
        
        while (x < nX) and (x >= 0) and (y < nY) and (y >= 0) and (z < nZ) and (z >= 0):
            
            vidx = int(x + nX * y + nX * nY * z)
            
            if (gvoxArr[vidx] == 1) or (gnd == 1):
                occlArr[vidx] += 1.0
                gnd = 1
                
            for i in range(number_of_returns):
                if (vidx == voxIdx[i]) and (gnd == 0):
                    hitsArr[vidx] += w
                    woccl += w
                    wmiss -= w
            
            if gnd == 0:
                occlArr[vidx] += woccl
                missArr[vidx] += wmiss 
                        
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
       tmin: min distance from the beam origin
       tmax: max distance from the beam origin
    """
    if dx != 0:
        divX = 1.0 / dx
    else:
        divX = 1.0
    
    if divX >= 0:
    	tmin = (bounds[0] - x0) * divX
    	tmax = (bounds[3] - x0) * divX
    else:
    	tmin = (bounds[3] - x0) * divX
    	tmax = (bounds[0] - x0) * divX
      
    if dy != 0:
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

        if dz != 0:
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
    
    return intersect,tmin,tmax
    
