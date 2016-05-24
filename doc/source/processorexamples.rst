==================
Processor Examples
==================

--------------
Simple Example
--------------

::

    # Reads in an input lidar file and writes an
    # output image file with the minimum 'Z' value for 
    # each bin
    import numpy
    from pylidar import lidarprocessor

    def writeImageFunc(data):
        # get a 3d array of data for this block
        zVals = data.input.getPointsByBins(colNames='Z')
        # determine the shape
        (maxPts, nRows, nCols) = zVals.shape
        nullval = 0
        if maxPts > 0:
            # there is data for this block - find minimum
            minZ = zVals.min(axis=0)
            # make 3d
            stack = numpy.ma.expand_dims(minZ, axis=0)
        else:
            # no data for this block - set to nullval
            stack = numpy.empty((1, nRows, nCols), dtype=zVals.dtype)
            stack.fill(nullval)

        # set result
        data.imageOut.setData(stack)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile('file.spd', lidarprocessor.READ)
    dataFiles.imageOut = lidarprocessor.ImageFile('outfile.img', lidarprocessor.CREATE)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles)

The program shown above is complete, and would work assuming file.spd existed and contained
a 'Z' value for each point. It would create an output raster file with the minimum Z value for each bin. 

The user-supplied function writeImageFunc is passed to the :func:`pylidar.lidarprocessor.doProcessing`
function which applies it accross the lidar file. By default PyLidar attempts to process a file
in a spatial manner. Inside the writeImageFunc function the 'data' object has attributes
named in the same way that the 'dataFiles' input to :func:`pylidar.lidarprocessor.doProcessing` has.
For inputs and outputs representing lidar files, objects of type :class:`pylidar.userclasses.LidarData`
are provided. For inputs and outputs representing raster files, objects of type
:class:`pylidar.userclasses.ImageData` are provided.

See the help on :class:`pylidar.lidarprocessor.DataFiles`, :class:`pylidar.lidarprocessor.LidarFile` and :class:`pylidar.lidarprocessor.ImageFile`
for more information on setting up inputs.

To understand more about the types of arrays returned by PyLidar for lidar data, 
see the :doc:`arrayvisualisation` page.

You can specify which columns you want returned from a lidar file by setting the colNames parameter
to getPoints (and other lidar reading functions). By default all columns are returned as a structured
array. If a list of columns is requested, this is also returned as a structured array. However if a
single column name is requested (as a string, not a list) then a non-structred array is returned.

Raster data is represented as 3-d numpy arrays. The first dimension corresponds to the number of layers in the image file, and will be present even when there is only one layer.
The datatype of the output file(s) will be inferred from the datatype of the numpy arrays(s) 
given to :func:`pylidar.userclasses.ImageData.setData`. So, to control the datatype of the output file, 
use the numpy astype() function to control the datatype of the output arrays.

See :func:`pylidar.lidarprocessor.setDefaultDrivers` for discussion of how to set the output GDAL driver.

-------------
Interpolation
-------------

Support for interpolation techniques is included with PyLidar. The following example shows how
to interpolate with the :func:`pylidar.toolbox.interpolation.interpGrid` function and also how
to filter by classification::

    from pylidar import lidarprocessor
    from pylidar.toolbox import interpolation

    def interpGroundReturns(data):
        # if given a list of fields, returns a structured array with all of them
        ptVals = data.input.getPoints(colsNames=['X', 'Y', 'Z', 'CLASSIFICATION'])
        # create mask for ground
        # TODO: update this when standard classifications are introduced
        mask = ptVals['CLASSIFICATION'] == 2

        # get the coords for this block
        pxlCoords = data.info.getBlockCoordArrays()

        if ptVals.shape[0] > 0:
            # there is data for this block
            xVals = ptVals['X'][mask]
            yVals = ptVals['Y'][mask]
            zVals = ptVals['Z'][mask]
            # 'pynn' needs the pynnterp module installed
            out = interpolation.interpGrid(xVals, yVals, zVals, pxlCoords, 'pynn')

            # mask out where interpolation failed
            invalid = numpy.isnan(out)
            out[invalid] = 0
        else:
            # no data - set to zero
            out = numpy.empty(pxlCoords[0].shape, dtype=numpy.float64)
            out.fill(0)

        data.imageOut.setData(out)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile('file.spd', lidarprocessor.READ)
    dataFiles.imageOut = lidarprocessor.ImageFile('outfile.img', lidarprocessor.CREATE)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles)


The data.info object is an instance of :class:`pylidar.userclasses.UserInfo` and contains
some useful functions for obtaining the current processing state.

---------------------------------------------
Arbitrary numbers of Input (and Output) Files
---------------------------------------------

Each name on the dataFiles object can also be a list of files, instead of a single file. 
This will cause the corresponding attribute on the dataFiles object to be a list also. 
This allows the function to process an arbitrary number of files, without having to give each one a separate name 
within the function. An example might be a function to to interpolate a DEM from many files, 
which should work the same regardless of how many files are to be input. This could be written as follows::

    from pylidar import lidarprocessor
    from pylidar.toolbox import interpolation

    def interpGroundReturns(data):
        # read all the files
        ptVals = [indata.getPoints(colNames=['X','Y','Z','CLASSIFICATION']) 
                for indata in data.allinputs]
        # turn into one big array
        ptVals = numpy.ma.hstack(ptVals)
        # create mask for ground
        # TODO: update this when standard classifications are introduced
        mask = ptVals['CLASSIFICATION'] == 2

        # get the coords for this block
        pxlCoords = data.info.getBlockCoordArrays()

        if ptVals.shape[0] > 0:
            # there is data for this block
            xVals = ptVals['X'][mask]
            yVals = ptVals['Y'][mask]
            zVals = ptVals['Z'][mask]
            # 'pynn' needs the pynnterp module installed
            out = interpolation.interpGrid(xVals, yVals, zVals, pxlCoords, 'pynn')

            # mask out where interpolation failed
            invalid = numpy.isnan(out)
            out[invalid] = 0
        else:
            # no data - set to zero
            out = numpy.empty(pxlCoords[0].shape, dtype=numpy.float64)
            out.fill(0)

        data.imageOut.setData(out)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.allinputs = []
    for infile in infiles:
        input = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
        dataFiles.allinputs.append(input)
    dataFiles.imageOut = lidarprocessor.ImageFile('outfile.img', lidarprocessor.CREATE)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles)

---------------------
Updating a Lidar File
---------------------

This example updates a Lidar file with data from an image raster::

    from pylidar import lidarprocessor

    def updatePointFunc(data):
        pts = data.input.getPointsByBins(colNames=['HEIGHT', 'Z'])
        (nPts, nRows, nCols) = pts.shape
        if nPts > 0:
            # read in the DEM data
            dem = data.imageIn.getData()
            # make it match the size of the pts array
            # ie repeat it for the number of bins
            dem = numpy.repeat(dem, pts.shape[0], axis=0)
            
            # calculate the height
            # ensure this is a masked array to match pts
            height = numpy.ma.array(pts['Z'] - dem, mask=pts['Z'].mask)
            pts['HEIGHT'] = height

            # update the lidar file
            data.input.setPoints(pts)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile('file.spd', lidarprocessor.UPDATE)
    dataFiles.imageIn = lidarprocessor.ImageFile('dem.img', lidarprocessor.READ)

    lidarprocessor.doProcessing(writeImageFunc, dataFiles)

If requesting a non-structured array like this::

    height = data.input.getPointsByBins(colNames='HEIGHT')

You will need to specify the colName when calling :func:`pylidar.userclasses.LidarData.setPoints`::

    data.input.setPoints(height, colName='HEIGHT')

---------------------
Notes for using Numba
---------------------

`Numba <http://numba.pydata.org/>`_ is a useful tool for doing processing that can't
be done by whole of array operations. However, Numba cannot currently deal with masked arrays.
A solution is to pass the "data" and "mask" attributes of your masked array separately 
to a Numba function.

--------------------------
Passing Other Data Example
--------------------------

A mechanism is provided for passing data other than lidar or raster data in and out of the
user function. This is obviously useful for passing parameters into the processing. It can also be used to pass information out again, and to preserve data between calls to the function, since the otherargs object is preserved between blocks.

When invoking :func:`pylidar.lidarprocessor.doProcessing` there is an optional named argument 'otherArgs'.
This can be any python object, but will typically be an instance of the :class:`pylidar.lidarprocessor.OtherArgs` class. 
If supplied, then the use function should also expect to take this as its fourth argument. It will be supplied to every call to the user function, and pylidar will do nothing to it between calls.

An example of finding the average 'Z' value accross a Lidar file (showing only relevant lines)::

    def findAverage(data, otherargs):
        zVals = data.input.getPoints(colNames='Z')
        otherargs.tot += zVals.sum()
        otherargs.count += zVals.shape[0]

    otherargs = lidarprocessor.OtherArgs()
    otherargs.tot = 0.0
    otherargs.count = 0
    lidarprocessor.doProcessing(findAverage, dataFiles, otherArgs=otherargs)
    print('Average Z', otherargs.tot / otherargs.count)

-----------------------------------
Controlling Reading/Writing Example
-----------------------------------

This example shows how to use the :class:`pylidar.lidarprocessor.Controls` class to restrict 
processing to the extent of another image file::

    controls = lidarprocessor.Controls()
    controls.setReferenceImage('footprint.img')
    controls.setFootprint(lidarprocessor.BOUNDS_FROM_REFERENCE)

    lidarprocessor.doProcessing(userFunc, dataFiles, controls=controls)

This example shows how to process data non-spatially (default is to process spatially)::

    controls = lidarprocessor.Controls()
    controls.setSpatialProcessing(False)

----------------------
Setting driver options
----------------------

Unlike GDAL and RIOS which only allowed setting of driver options on file creation,
PyLidar supports setting of driver options on reading. This example shows how to set the
BIN_SIZE option for LAS files which need this set before they can read data spatially::

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile('file.spd', lidarprocessor.UPDATE)
    dataFiles.input.setLiDARDriverOption('BIN_SIZE, 1.0)
