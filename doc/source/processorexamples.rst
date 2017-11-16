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
    from numba import jit
    from pylidar import lidarprocessor
    from pylidar.toolbox import spatial
    from pylidar.lidarformats import generic

    BINSIZE = 1.0

    @jit
    def findMinZs(data, outImage, xMin, yMax):
        for i in range(data.shape[0]):
            if data[i]['CLASSIFICATION'] == lidarprocessor.CLASSIFICATION_GROUND:
                row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                        xMin, yMax, BINSIZE)
                if outImage[row, col] != 0:
                    if data[i]['Z'] < outImage[row, col]:
                        outImage[row, col] = data[i]['Z']
                else:
                    outImage[row, col] = data[i]['Z']

    def processChunk(data, otherArgs):
        lidar = data.input1.getPoints(colNames=['X', 'Y', 'Z', 'CLASSIFICATION'])
        findMinZs(lidar, otherArgs.outImage, otherArgs.xMin, otherArgs.yMax)

    info = generic.getLidarFileInfo(inFile)
    header = info.header

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(inFile, lidarprocessor.READ)

    xMin, yMax, ncols, nrows = spatial.getGridInfoFromHeader(header, BINSIZE)

    outImage = numpy.zeros((nrows, ncols))

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.outImage = outImage
    otherArgs.xMin = xMin
    otherArgs.yMax = yMax

    lidarprocessor.doProcessing(processChunk, dataFiles, otherArgs=otherArgs)

    iw = spatial.ImageWriter(outFile, tlx=xMin, tly=yMax, binSize=BINSIZE)
    iw.setLayer(outImage)
    iw.close()

The program shown above is complete, and would work assuming inFile existed and contained
a 'Z' value for each point. It would create an output raster file with the minimum Z value for each bin. 

The user-supplied function processChunk is passed to the :func:`pylidar.lidarprocessor.doProcessing`
function which applies it through the lidar file in chunks. Inside the processChunk function the 'data' object has attributes
named in the same way that the 'dataFiles' input to :func:`pylidar.lidarprocessor.doProcessing` has. Each object
is of type :class:`pylidar.userclasses.LidarData`.

See the help on :class:`pylidar.lidarprocessor.DataFiles`, :class:`pylidar.lidarprocessor.LidarFile`
and :class:`pylidar.lidarprocessor.OtherArgs` for more information on setting up inputs. There is a section below
which talks about other inputs in more detail.

You can specify which columns you want returned from a lidar file by setting the colNames parameter
to getPoints (and other lidar reading functions). By default all columns are returned as a structured
array. If a list of columns is requested, this is also returned as a structured array. However if a
single column name is requested (as a string, not a list) then a non-structred array is returned.

The data.info object is an instance of :class:`pylidar.userclasses.UserInfo` and contains
some useful functions for obtaining the current processing state.

The :mod:`pylidar.toolbox.spatial` module has functions and classes to assist processing LiDAR data
spatially.

See :func:`pylidar.lidarprocessor.setDefaultDrivers` for discussion of how to set the output GDAL driver.

-------------
Interpolation
-------------

Support for interpolation techniques is included with PyLidar. Note that all the data for the area to be
interpolated needs too be read in - the :func:`pylidar.toolbox.spatial.readLidarPoints` makes
this easier. The following example shows how
to interpolate with the :func:`pylidar.toolbox.interpolation.interpGrid` function and also how
to filter by classification::

    from pylidar import lidarprocessor
    from pylidar.toolbox import spatial
    from pylidar.toolbox import interpolation

    BINSIZE = 1.0

    data = spatial.readLidarPoints(inFile, 
            classification=lidarprocessor.CLASSIFICATION_GROUND)

    (xMin, yMax, ncols, nrows) = spatial.getGridInfoFromData(data['X'], data['Y'],
                BINSIZE)

    pxlCoords = spatial.getBlockCoordArrays(xMin, yMax, ncols, nrows, BINSIZE)

    dem = interpolation.interpGrid(data['X'], data['Y'], data['Z'], pxlCoords, 'pynn') 

    iw = spatial.ImageWriter(outFile, tlx=xMin, tly=yMax, binSize=BINSIZE)
    iw.setLayer(dem)
    iw.close()

---------------------------------------------
Arbitrary numbers of Input (and Output) Files
---------------------------------------------

Each name on the dataFiles object can also be a list of files, instead of a single file. 
This will cause the corresponding attribute on the dataFiles object to be a list also. 
This allows the function to process an arbitrary number of files, without having to give each one a separate name 
within the function. An example might be a function to find the minimum Z from many files, 
which should work the same regardless of how many files are to be input. This could be written as follows::

    import numpy
    from numba import jit
    from pylidar import lidarprocessor
    from pylidar.toolbox import spatial
    from pylidar.lidarformats import generic

    BINSIZE = 1.0

    @jit
    def findMinZs(data, outImage, xMin, yMax):
        for i in range(data.shape[0]):
            if data[i]['CLASSIFICATION'] == lidarprocessor.CLASSIFICATION_GROUND:
                row, col = spatial.xyToRowColNumba(data[i]['X'], data[i]['Y'],
                        xMin, yMax, BINSIZE)
                if outImage[row, col] != 0:
                    if data[i]['Z'] < outImage[row, col]:
                        outImage[row, col] = data[i]['Z']
                else:
                    outImage[row, col] = data[i]['Z']

    def processChunk(data, otherArgs):
    
        for input in data.allinputs:
            lidar = input.getPoints(colNames=['X', 'Y', 'Z', 'CLASSIFICATION'])
            findMinZs(lidar, otherArgs.outImage, otherArgs.xMin, otherArgs.yMax)

    headers = []
    for inFile in inFiles:
        info = generic.getLidarFileInfo(inFile)
        headers.append(info.header)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.allinputs = []
    for inFile in inFiles:
        inp = lidarprocessor.LidarFile(inFile, lidarprocessor.READ)
        dataFiles.allinputs.append(inp)

    xMin, yMax, ncols, nrows = spatial.getGridInfoFromHeader(headers, BINSIZE)
    outImage = numpy.zeros((nrows, ncols))

    otherArgs = lidarprocessor.OtherArgs()
    otherArgs.outImage = outImage
    otherArgs.xMin = xMin
    otherArgs.yMax = yMax

    lidarprocessor.doProcessing(processChunk, dataFiles, otherArgs=otherArgs)

    iw = spatial.ImageWriter(imageFile, tlx=xMin, tly=yMax, binSize=BINSIZE)
    iw.setLayer(outImage)
    iw.close()


This assume that inFiles is a list.

---------------------
Updating a Lidar File
---------------------

This example updates a Lidar file by creating a new column with data from an image raster::

    import numpy
    from numba import jit
    from pylidar import lidarprocessor
    from pylidar.toolbox import spatial
    from pylidar.toolbox import arrayutils

    def processChunk(data, otherArgs):
        lidar = data.input1.getPoints(colNames=['X', 'Y', 'Z'])
        rows, cols = spatial.xyToRowCol(lidar['X'], lidar['Y'], 
                    otherArgs.xMin, otherArgs.yMax, otherArgs.binSize)

        height = lidar['Z'] - otherArgs.inImage[rows, cols]
        lidar = arrayutils.addFieldToStructArray(lidar, 'HEIGHT', numpy.float, height)
        data.input1.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, 10, -10)
        data.input1.setPoints(lidar)

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(lidarFile, lidarprocessor.UPDATE)

    otherArgs = lidarprocessor.OtherArgs()
    (otherArgs.inImage, otherArgs.xMin, otherArgs.yMax, otherArgs.binSize) = spatial.readImageLayer(imageFile)

    lidarprocessor.doProcessing(processChunk, dataFiles, otherArgs=otherArgs)

You can also update 'in-place' by changing the values in an existing column before calling setPoints().

If requesting a non-structured array like this::

    height = data.input.getPoints(colNames='HEIGHT')

You will need to specify the colName when calling :func:`pylidar.userclasses.LidarData.setPoints`::

    data.input.setPoints(height, colName='HEIGHT')

New columns can be created in SPDV4 format by creating a new column in the structured
array passed to :func:`pylidar.userclasses.LidarData.setPoints`, or a new colName for 
non-structured arrays.

---------------------
Reading Waveform Data
---------------------

Reading waveform data is very similar to reading points and pulses. Here is a simple
example::

    def readFunc(data):
        # returns 2d masked structured array with info about waveforms
        # first axis is waveform number, second is pulse
        waveinfo = data.input1.getWaveformInfo()

        # returns masked 3d radiance array
        # first axis is waveform bin, second is waveform number, third is pulse
        recv = data.input1.getReceived()
        trans = data.input1.getTransmitted()

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    lidarprocessor.doProcessing(readFunc, dataFiles)

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

This example shows how to use the :class:`pylidar.lidarprocessor.Controls` class to change the 
size of the chunk of lidar data read::

    controls = lidarprocessor.Controls()
    controls.setWindowSize(256) # actually uses 256*256 for legacy reasons...

    lidarprocessor.doProcessing(userFunc, dataFiles, controls=controls)

----------------------
Setting driver options
----------------------

Unlike GDAL and RIOS which only allowed setting of driver options on file creation,
PyLidar supports setting of driver options on reading. This example shows how to set the
BIN_SIZE option for LAS files which need this set before they can read data spatially::

    dataFiles = lidarprocessor.DataFiles()
    dataFiles.input = lidarprocessor.LidarFile('file.spd', lidarprocessor.UPDATE)
    dataFiles.input.setLiDARDriverOption('BIN_SIZE', 1.0)

------------------------
Accessing SPDV4 features
------------------------

When using SPDV4, it might be useful to use these features:

* Getting and setting scaling for columns - see :func:`pylidar.userclasses.LidarData.getScaling` and :func:`pylidar.userclasses.LidarData.setScaling`.
* Getting and setting the native data type for columns - see :func:`pylidar.userclasses.LidarData.getNativeDataType` and :func:`pylidar.userclasses.LidarData.setNativeDataType`.
* Getting and setting the null value for columns - see :func:`pylidar.userclasses.LidarData.getNullValue` and :func:`pylidar.userclasses.LidarData.setNullValue`.
