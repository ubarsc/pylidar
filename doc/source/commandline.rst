=====================
Command Line Examples
=====================

----------------------------------
Translation with pylidar_translate
----------------------------------

Run "pylidar_translate -h" to obtain full help. The basic usage is to specify input and output files
plus the output format like this::
    
    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4

^^^^^^^^^^^^^^^^^^^
Setting the scaling
^^^^^^^^^^^^^^^^^^^

When writing to SPDV4 files, scaling can be set on the output columns with the --scaling option. This
can be specified multiple times, once for each column that you need to set the scaling on. The
--scaling option takes 4 parameters. The first is a string describing what sort of column it is and should
be one of POINT, PULSE or WAVEFORM. The second is the column name. The third is the data type and can be one of the 
following strings: INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64 or DFLT.
'DFLT' means that the default type for the given column is used and it is assumed that the column is one
of the recognised ones from :doc:`spdv4format`. The fourth and fifth arguments are the gain and offset respectively 
and are interpreted as floats. Here is an example or overridding the scaling on the 'Z' column::

    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4 --scaling POINT Z UINT32 1 -500

Scaling is applied when saving data to a file using the expression::

    (data - offset) * gain

And un-applied when reading from a file using the expression::

    (data / gain) + offset

^^^^^^^^^^^^^^^^^^^^^^
Setting the null value
^^^^^^^^^^^^^^^^^^^^^^

When writing to SPDV4 files, the null value can be set on the output columns with the --null option. This
can be specified multiple times, once for each column that you need to set the scaling on. The --null
option takes 4 parameters. The first describes the sort of column (POINT, PULSE or WAVEFORM). The
second is the column name and the third is the actual null value. Note that scaling values (if set)
will be applied to this value. Here is an example of setting the null value for the 'Z' column to 0::

    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4 --null POINT Z 0
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Checking the expected range
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some circumstances it is useful to check that the data is within an expected range before proceeding. This 
can be accomplished using the --range option to specify columns and their ranges. If the data is outside the 
given ranges, or the column(s) do not exist then an error is raised. The --range option takes 4 arguments. The
first is a string describing what sort of column it is and should be one of POINT, PULSE or WAVEFORM. The second is the column name.
The third and fourth are the expected minimum and maximum values respectively. Here is an example::

    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4 --range POINT Z 0 50

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Spatial Processing (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For input formats that have a spatial index, the data can be processed in a spatial way and a new spatial index calculated 
in the output file. The default is to process the data non-spatially. To enable spatial processing, use the --spatial option.

You may also need to set the --binsize flag for LAS files.

You can also restrict spatial processing to a certain extent using the --extent flag. This takes four options:
xmin ymin xmax ymax. This is an example of spatial processing restricted to a certain extent::

    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4 --spatial --binsize 1 --extent 664500 7765999 664999 7767000

^^^^^^^^^^^^^^^^
Constant Columns
^^^^^^^^^^^^^^^^

When creating an SPDV4 file extra columns can be set in the data that wasn't present in the 
original file. These columns are initialised with a constant value. This can be useful
when an input file is has all the same CLASSIFICATION or RETURN_TYPE and this needs to 
be shown in the output when such a column doesn't exist in the input. The --constcol option takes 4 arguments. The
first is a string describing what sort of column it is and should be one of POINT, PULSE or WAVEFORM. The second is the column name.
The third is the data type and can be one of the following strings: INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64.
Here is an example that creates a 'SOURCE' column for points that is UINT16 and initialised to 39::

    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4 --constcol POINT SOURCE UINT16 39

^^^^^^^^^^^^^^^^^^^^
LAS Specific Options
^^^^^^^^^^^^^^^^^^^^

The --binsize is required for input LAS files when processing spatially and this sets the size of each spatial bin. --epsg
allows you to set the projection explicity since many LAS files do not have the projection set. --buildpulses tells
pylidar to build a pulse structure from the input. If this isn't specified then there is one pulse for each point.
--pulseindex allows to you to specify either FIRST_RETURN or LAST_RETURN and this dictates how the points are indexed
to the pulses.

^^^^^^^^^^^^^^^^^^^^^^^^^^
RIEGL RXP Specific Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

--internalrotation tells pylidar to use the internal instrument inclinometer and compass data within the file to transform 
point and pulse coordinates. --externalrotationfn allows the user to specify the transform in a text file. --magneticdeclination 
allows the user to specify a number for the magnetic declination that needs to be corrected when using the internal 
instrument compass data. 

^^^^^^^^^^^^^^^^^^^^^^
ASCII Specific Options
^^^^^^^^^^^^^^^^^^^^^^

For ASCII Formats you need to specify all the columns and their types in the input file. This can be done with the --coltype
option multiple times, once for each column. This takes two parameters, first is the column name, second is the data type. 
The data type can be one of: INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT32, FLOAT64. 
For Time Sequential data, you also need to specify with columns are pulse columns with the --pulsecols. 
This needs to be a comma separated
list of column names that are to be treated as pulses. Here is an example::

    pylidar_translate --input data.dat.gz --output data.spdv4 --format SPDV4 \
        --coltype GPS_TIME FLOAT64 --coltype X_IDX FLOAT64 --coltype Y_IDX FLOAT64 \
        --coltype Z_IDX FLOAT64 --coltype X FLOAT64 --coltype Y FLOAT64 --coltype Z FLOAT64 \
        --coltype CLASSIFICATION UINT8 --coltype ORIG_RETURN_NUMBER UINT8 \
        --coltype ORIG_NUMBER_OF_RETURNS UINT8 --coltype AMPLITUDE FLOAT64 \
        --coltype FWHM FLOAT64 --coltype RANGE FLOAT64 --pulsecols GPS_TIME,X_IDX,Y_IDX,Z_IDX

You may also need to translate from internal codes used the CLASSIFICATION column to standard codes.
Use the --classtrans option for this (see "pylidar_translate -h" for more information)::

    --classtrans 5 INSULATOR --classtrans 6 HIGHVEGE

---------------------------------------------------------------------
Creating a raster from LiDAR data with pylidar_rasterize (deprecated)
---------------------------------------------------------------------

The pylidar_rasterize command takes one or more input LiDAR files and creates a raster
from it. You need to specify the attribute name(s) to use. An output layer is created for 
each attribute. My default a minimum function is used to turn the data for a bin into a raster
value, but other functions can be used as long as they accept a 
`Masked Array <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ as the first
parameter and the "axis" and "fill_value" parameters. Functions in the "numpy.ma" module
are a good starting point.

This example creates a raster using the minimum 'Z'
values in a raster::

    pylidar_rasterize -i data.spd -o minz.img -a Z

Multiple attributes can be specified, or another function::

    pylidar_rasterize -i data.spd -o maxz.img -a Z X -f numpy.ma.max

By default, point data is processed, but this can be changed to pulses with the --type
option::

    pylidar_rasterize -i data.spd -o minx.img -a X_IDX -t PULSE
    
-----------------------------------------------
Getting Information on a File with pylidar_info
-----------------------------------------------

The pylidar_info command takes a --input option to specify the path to a file. Information about the file
is then printed on the terminal. Here is an example::

    pylidar_info --input data.rxp
    
--------------------------------------------------------
Creating a Spatial Index with pylidar_index (deprecated)
--------------------------------------------------------

Once you have converted your data to SPV4 format, you can use this
utility to spatially index the data. Once it is spatially indexed, 
you can process it spatially.

Run "pylidar_index -h" to obtain full help. The basic usage is to specify input and output files
like this::
    
    pylidar_index --input data.spdv4 --output indexed.spdv4

Important options that you should consider overriding are:

    * The resolution that the spatial index is calculated on (the -r or --resolution flag). This will control how much data goes in each bin.
    * The type of index (the --indextype flag). This controls what the spatial index is calculated on. CARTESIAN or SCAN is usual for airborne, but SPHERICAL for TLS.
    * The size of the blocks that the spatial indexing will use (the -b or --blocksize flag).This will determine the amount of memory used. The smaller the blocks the less memory will be used. The value will be in the units that the spatial index is being calculated in. By default pylidar_index uses 200.
    * The temporary directory to create the temprary files in. By default this is the current directory, but you may want to change this if you do not have enough space there.

-----------------------------------------------
Splitting a File into Tiles using pylidar_tiles
-----------------------------------------------

Once you have converted your data to SPDV4 format, you can use this
utility to split it into tiles so they can be processed independently.

Run "pylidar_tile -h" to obtain full help. The basic usage is to specify a input file like this::

    pylidar_tile --input data.spdv4

Many of the flags are similar to "pylidar_index" so consult the help above for more information on these.


--------------------------------------------
Deriving canopy metrics using pylidar_canopy
--------------------------------------------

Once you have converted your data to SPDV4 format, you can use this
utility to derive various published lidar canopy metrics. Some metrics will also 
accept other suitable file formats.

Currently only vertical plant profiles [Pgap(theta,z), PAI(z), PAVD(z)] from 
TLS as described by Calders et al. (2014) Agricultural and Forest 
Meteorology are implemented. These are designed to stratify gap fraction, 
plant area index, and plant area volume density by height when only
single scan locations are measured. RXP and SPD files are accepted as input.

Run "pylidar_canopy -h" to obtain full help. The basic usage is to specify 
1 or 2 (in the case of a RIEGL VZ400) input files like this::

    pylidar_canopy -i tls_scan_upright.spd tls_scan_tilted.spd -o vertical_profiles.csv
        --minzenith 35.0 5.0 --maxzenith 70.0 35.0 --heightcol HEIGHT

    pylidar_canopy -i tls_scan_upright.rxp -o vertical_profiles.csv
        -p -r planefit.rpt --minzenith 35.0 --maxzenith 70.0


The output is a CSV file with a table of vertical profile metrics [Pgap(theta,z), PAI(z), PAVD(z)] 
as columns and vertical height bin (m) starting points as rows. This command can also apply a 
plane fit on-the-fly to correct a single location scan for topographic effects (the -p option) 
and also output a report on the fit statistics (the -r option). If point heights are already 
defined in an SPD file (e.g. from a DEM), specify the point column name to use with --heightcol.

In the above example, only view zenith angles between 35 and 70 degrees are used for the 
tls_scan_upright.spd file and 5 and 35 degrees for the tls_scan_tilted.spd file. These are 
recommended values for the RIEGL VZ400 (Calders et al., 2014).




