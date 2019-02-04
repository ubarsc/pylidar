"""
Functions for multilayer crown delineation from ALS point cloud data (Duncanson et al., 2014)
Assumes a ground classification exists and is included in the input file format.
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
import pynninterp
from numba import jit
from scipy import ndimage

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.morphology import disk

from pylidar.toolbox import spatial
from pylidar.lidarformats import generic


def writeTestImage(fn,image,otherargs):
    """
    Temporary outputs for debugging
    """
    outImageFile = spatial.ImageWriter(fn, tlx=otherargs.bounds[0], tly=otherargs.bounds[3], 
            binSize=otherargs.resolution, driverName=otherargs.rasterdriver, epsg=otherargs.proj[0], 
            numBands=1) 
    outImageFile.setLayer(image, layerNum=1)    
    outImageFile.close()
    

def run_crown_duncanson2014(points, otherargs, outfiles):
    """
    Main function for CROWN_DUNCANSON2014
    """
    
    # Initialize the output files
    outImageFile = spatial.ImageWriter(outfiles[0], tlx=otherargs.bounds[0], tly=otherargs.bounds[3], 
            binSize=otherargs.resolution, driverName=otherargs.rasterdriver, epsg=otherargs.proj[0], 
            numBands=otherargs.maxlayers, nullVal=0)  
    
    # Output grid size
    nX = int((otherargs.bounds[1] - otherargs.bounds[0]) / otherargs.resolution)
    nY = int((otherargs.bounds[3] - otherargs.bounds[2]) / otherargs.resolution)
    
    # Calculate height above ground of all points
    heights = calcHeights(points)
    
    # Run the process for a user defined number of layers
    for layer in range(otherargs.maxlayers):
    
        # Initialize the output grid
        outImage = numpy.zeros((nY,nX))
    
        if points.shape[0] > 0:
        
            # Derive the regions for the entire point cloud
            regions = processLayer(points, heights, None, outImage, otherargs)
            
            # Process vertical profiles for each region
            points_mask,region_ids,region_heights = processProfiles(regions, points, heights, otherargs)
            
            # Derive the regions for the upppermost layer point cloud
            regions = processLayer(points, heights, points_mask, outImage, otherargs)
            
            # Write to the output image stack
            outImageFile.setLayer(regions, layerNum=layer+1)
            
            # Get the crown metrics for the uppermost layer
            metrics_tmp = extractMetrics(regions, region_ids, region_heights, layer+1, otherargs)
            metrics = numpy.concatenate((metrics, metrics_tmp)) if layer > 0 else metrics_tmp
            
            # Remove the upppermost layer from the point cloud
            points = points[~points_mask]
            heights = heights[~points_mask]
            
    outImageFile.close()
    numpy.savetxt(outfiles[1], metrics, fmt="%s", delimiter=',', 
            header=','.join(metrics.dtype.names))
        

def processLayer(points, heights, points_mask, outImage, otherargs):
    """
    Segment the uppermost canopy layer
    """        
    # Get the canopy height model
    if points_mask is None:
        findMaxHeights(points, heights, outImage, otherargs.bounds[0], otherargs.bounds[3], 
                otherargs.resolution)
    else:
        findMaxHeights(points[points_mask], heights[points_mask], outImage, 
                otherargs.bounds[0], otherargs.bounds[3], otherargs.resolution)
    
    # Fill pits in the CHM
    filled_chm = fillCHM(outImage, windowsize=otherargs.windowsize, minheight=otherargs.minheight)
    
    # Smooth the filled CHM
    smoothed_chm = smoothCHM(filled_chm, windowsize=otherargs.windowsize, 
            minheight=otherargs.minheight)
    
    # Run the watershed
    regions = runWatershed(smoothed_chm, windowsize=otherargs.windowsize, 
            minheight=otherargs.minheight)    
    
    return regions

    
def processProfiles(regions, points, heights, otherargs):
    """
    Extract the vertical profiles for the current set of segments
    and mask points that are not in the uppermost canopy
    """        
    # Initialize the profiles array
    region_ids,region_cnts = numpy.unique(regions, return_counts=True)
    z_max = numpy.max(heights)
    nz = int(z_max / otherargs.heightbinsize) + 1
    profile_data = numpy.zeros((region_ids.size, nz))
    profile_heights = numpy.zeros(region_ids.size)
    
    # Derive the vertical profile for each segment  
    binHeightsByRegion(points, heights, regions, region_ids, profile_data, profile_heights,  
            otherargs.bounds[0], otherargs.bounds[3], otherargs.resolution, otherargs.heightbinsize)
    
    # Segment the profiles
    height_thresholds = numpy.zeros(region_ids.size)
    segmentProfiles(profile_data, profile_heights, height_thresholds, otherargs.heightbinsize)
        
    # Update the point index
    points_mask = numpy.full(heights.size, False)
    maskPoints(points, heights, regions, region_ids, height_thresholds, points_mask, 
            otherargs.bounds[0], otherargs.bounds[3], otherargs.resolution)
    
    return points_mask,region_ids,profile_heights


@jit(nopython=True)
def binHeightsByRegion(data, heights, region_image, region_ids, profile_data, profile_heights,
    xMin, yMax, binSize, heightBinSize):
    """
    Create a maximum canopy height grid
    """
    for i in range(data.shape[0]):
        row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                xMin, yMax, binSize)
        if (row >= 0) & (col >= 0) & (row < region_image.shape[0]) & (col < region_image.shape[1]):
            if region_image[row, col] != 0:
                for j in range(region_ids.shape[0]):
                    if region_ids[j] == region_image[row, col]:
                        if heights[i] > profile_heights[j]:
                            profile_heights[j] = heights[i]
                        hbin = int(heights[i] / heightBinSize)
                        if (hbin >= 0) & (hbin < profile_data.shape[1]):
                            profile_data[j,hbin] += 1


#@jit(nopython=True)
def segmentProfiles(profile_data, profile_heights, height_thresholds, heightbinsize, 
    vertical_buffer=0.1, relative_thres=0.15):
    """
    Smooth and split the vertical profile
    Smoothing width is dependent on height of the tree, 0.15*segment height in bins
    Buffer is 10% of the max height (so passing a truncated profile)
    """
    peak_idx = 0
    height_bins = numpy.arange(profile_data.shape[0]) * heightbinsize
    for i in range(profile_heights.shape[0]):
        if profile_heights[i] > 0:
            smoothing_width = int(relative_thres * profile_heights[i])   
            if smoothing_width > 0:
                start_bin = int(profile_heights[i] * vertical_buffer / heightbinsize)        
                end_bin = int(profile_heights[i] / heightbinsize) - start_bin    
                for_dip = profile_data[i,start_bin:end_bin]
                if numpy.max(for_dip) > 0:
                    bins_dip = height_bins[start_bin:end_bin]
                    findProfilePeak(for_dip, bins_dip, peak_idx, smoothing_width=smoothing_width)
                    if peak_idx > 0:
                        height_thresholds[i] = bins_dip[peak_idx]
    

@jit(nopython=True)
def maskPoints(data, heights, region_image, region_ids, height_thresholds, data_mask, xMin, yMax, binSize):
    """
    Mask points that are not in the uppermost canopy
    """
    for i in range(data.shape[0]):
        row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                xMin, yMax, binSize)
        if (row >= 0) & (col >= 0) & (row < region_image.shape[0]) & (col < region_image.shape[1]):
            if region_image[row, col] != 0:
                for j in range(region_ids.shape[0]):
                    if region_ids[j] == region_image[row, col]:
                        if heights[i] > height_thresholds[j]:
                            data_mask[i] = True                

                
def extractMetrics(regions, region_ids, region_heights, layer, otherargs):
    """
    Extract metrics for each segment
    Implemented as done in Duncanson et al. (2014) but can be improved
    """    
    # Generate easting and northing values for each pixel
    xgrid,ygrid = numpy.meshgrid(numpy.arange(regions.shape[1]),numpy.arange(regions.shape[0]))
    xgrid = xgrid * otherargs.resolution + otherargs.bounds[0]
    ygrid = otherargs.bounds[3] - ygrid * otherargs.resolution
    
    # Initialize the output metrics structured array   
    arrayDtype = [("CrownID", 'i4'),("Layer", 'i4')]
    for name in ["Easting", "Northing", "Height", "Area", "DiameterX", "DiameterY"]:
        arrayDtype.append((name, 'f8'))
    metrics = numpy.zeros((region_ids.shape[0], ), dtype=arrayDtype)
    
    # Calculate the metrics
    metrics["CrownID"] = region_ids
    metrics["Layer"] = layer
    metrics["Height"] = region_heights    
    for i,region_id in enumerate(region_ids):
        if region_heights[i] > 0:
            idx = regions == region_id
            if idx.sum() > 0:
                metrics["Easting"][i] = numpy.mean(xgrid[idx])
                metrics["Northing"][i] = numpy.mean(ygrid[idx])       
                metrics["Area"][i] = numpy.sum(idx) * otherargs.resolution**2
                metrics["DiameterX"][i] = numpy.max(xgrid[idx]) - numpy.min(xgrid[idx])
                metrics["DiameterY"][i] = numpy.max(ygrid[idx]) - numpy.min(ygrid[idx])
    
    return metrics[metrics["Height"] > 0]
    
    
def calcHeights(data):
    """
    Calculate canopy heights for all points
    Natural neighbour interpoliation is used to estimate ground elevation
    underneath nonground returns
    """
    # Interpolate the ground elevation of nonground points
    gndmask = data['CLASSIFICATION'] == generic.CLASSIFICATION_GROUND
    nongndxy = numpy.empty((numpy.count_nonzero(~gndmask),2))
    nongndxy[:,0] = data['X'][~gndmask]
    nongndxy[:,1] = data['Y'][~gndmask]
    gndz = pynninterp.NaturalNeighbourPts(data['X'][gndmask], data['Y'][gndmask], 
            data['Z'][gndmask], nongndxy)
    
    # Calculate the heights    
    heights = numpy.zeros(data.shape[0], dtype=numpy.float32)
    heights[~gndmask] = data['Z'][~gndmask] - gndz
    
    # Exclude canopy points outside bounds of ground points
    # Need to verify why the NaN are occuring
    heights[numpy.isnan(heights)] = 0.0
    
    return heights
    

@jit(nopython=True)
def findMaxHeights(data, heights, outImage, xMin, yMax, binSize):
    """
    Create a maximum canopy height grid
    """
    for i in range(data.shape[0]):
        row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                xMin, yMax, binSize)
        if (row >= 0) & (col >= 0) & (row < outImage.shape[0]) & (col < outImage.shape[1]):
            if outImage[row, col] != 0:
                if heights[i] < outImage[row, col]:
                    outImage[row, col] = heights[i]
            else:
                outImage[row, col] = heights[i]

                
def fillCHM(inImage, windowsize=3, minheight=2.0):
    """
    Fill within canopy gaps with the mean of surrounding canopy pixels heights
    Within canopy gaps are defined as pixels with a height less than a user defined
    threshold, but surrounded by at least 5 canopy heights greater than the threshold.
    The infill value is calculated as the mean value of canopy pixels. The neighborhood
    is defined by windowsize * 2 + 1
    """
    # Calculate the median heights
    offset = 0
    median_heights = ndimage.median_filter(inImage, size=windowsize, mode='mirror', 
            origin=[offset,offset])
    
    # Generate inputs to calculate the mean heights of canopy pixels
    weights = numpy.ones([windowsize,windowsize])
    tmp_heights = (inImage >= minheight).astype(inImage.dtype)
    cnt_heights = ndimage.convolve(tmp_heights, weights=weights, mode='mirror', 
            origin=[offset,offset])
    tmp_heights = numpy.where(inImage >= minheight, inImage, 0.0)
    sum_heights = ndimage.convolve(tmp_heights, weights=weights, mode='mirror', 
            origin=[offset,offset])
    
    # Infill the CHM
    mask = (inImage < minheight) & (median_heights >= minheight)
    avg_heights = numpy.divide(sum_heights, cnt_heights, where=mask)
    filled_chm = numpy.where(mask, avg_heights, inImage)
    
    return filled_chm

    
def smoothCHM(inImage, windowsize=3, minheight=2.0):
    """
    Smooth the canopy height model. The neighborhood
    is defined by windowsize * 2 + 1
    """
    # Set the disk structuring element
    r = (windowsize - 1) / 2
    d = disk(r)
    w = d / numpy.sum(d)
    
    # Smooth the CHM    
    tmp = numpy.where(inImage >= minheight, inImage, 0.0)
    smoothed_chm = ndimage.convolve(tmp, weights=w, 
            mode='mirror', origin=[0,0])
    
    return smoothed_chm

    
def runWatershed(inImage, windowsize=3, minheight=2.0):
    """
    Run the scikit learn watershed algorithm
    Use local maximum peaks as input markers
    """
    # Generate the input markers
    d = numpy.ones((windowsize,windowsize))
    local_maxi = peak_local_max(inImage, threshold_abs=minheight,  
            footprint=d, indices=False)
    markers = ndimage.label(local_maxi)[0]
    
    # Run the watershed
    min_val = numpy.min(inImage)
    max_val = numpy.max(inImage)   
    val_range = max_val - min_val
    tmpImage = numpy.where(val_range > 0, (inImage - min_val) / val_range, 0)
    tmpImage = max_val - (255 + 0.9999) * tmpImage
    
    canopy_mask = inImage >= minheight
    labels = watershed(tmpImage, markers, mask=canopy_mask)
    
    return labels

    
#@jit(nopython=True)
def findProfilePeak(for_dip, bins_dip, peak_arr_x2, n_down=4, w_siz=2, second_peak_th_factor=0.06, smoothing_width=4):
    """
    Matthew Brolly's peak finding code
    for_dip = the profile
    bins_dip = the height bins
    peak_arr_x2 = the output array
    """
    normal_input = 1 - (for_dip / numpy.max(for_dip))
    k = numpy.ones(smoothing_width) / smoothing_width
    temp_var = numpy.convolve(normal_input, k)    
    max_peak = numpy.max(temp_var)
    max_peak_x, = numpy.where(temp_var == max_peak)
    peak_th = second_peak_th_factor * max_peak
    peak_arr = numpy.zeros(temp_var.size)
    peak_arr_x = numpy.zeros(temp_var.size,dtype=numpy.uint16)
    previous_w_peak = 0
    peak_assignment_flag = 0
    i = 0
    for x in range(0, temp_var.size-w_siz):
        
        current_w_peak = numpy.max(temp_var[x:x+w_siz])
        
        if ((current_w_peak >= previous_w_peak) and (current_w_peak > peak_th)):
            previous_w_peak = current_w_peak
            previous_peak = current_w_peak
            previous_peak_x_tmp, = numpy.where(temp_var[x:x+w_siz] == current_w_peak)
            previous_peak_x = x + previous_peak_x_tmp[0]
            peak_assignment_flag = 0
           
        if current_w_peak < previous_w_peak:
            previous_w_peak = current_w_peak
            peak_assignment_flag += 1

        if peak_assignment_flag == n_down:
            peak_arr[i] = previous_peak
            peak_arr_x[i] = previous_peak_x
            i += 1

        previous_w_peak = current_w_peak

    selected_i, = numpy.where(peak_arr > 0)
    
    if selected_i.size > 0:
        peak_arr_x2 = peak_arr_x[selected_i][0]
    else:
        peak_arr_x2 = -1
