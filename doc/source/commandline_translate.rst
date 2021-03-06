========================================
Command Line Examples: pylidar_translate
========================================

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


