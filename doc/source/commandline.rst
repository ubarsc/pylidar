=====================
Command Line Examples
=====================

More to come::

    # LAS to SPDV4
    pylidar_translate --input data.laz --output data.spdv4 --format SPDV4 --epsg 28356 --scaling POINT Z UINT32 1 -500 --spatial --binsize 1

    # SPDV3 to SPDV4
    pylidar_translate --input data.spd --output data.spdv4 --format SPDV4 --spatial

    # Riegl to SPDV4 with override of scaling
    pylidar_translate --input data.rxp --output data.spdv4 --format SPDV4 --scaling PULSE Y_ORIGIN INT32 1.0 100 

    # SPDV4 to LAS
    pylidar_translate --input data.spdv4 --output data.las --format LAS

    # ASCII to SPDV4
    pylidar_translate --input data.dat.gz --output data.spdv4 --format SPDV4 \
        --coltype GPS_TIME FLOAT64 --coltype X_IDX FLOAT64 --coltype Y_IDX FLOAT64 
        --coltype Z_IDX FLOAT64 --coltype X FLOAT64 --coltype Y FLOAT64 --coltype Z FLOAT64 \
        --coltype CLASSIFICATION UINT8 --coltype ORIG_RETURN_NUMBER UINT8 \
        --coltype ORIG_NUMBER_OF_RETURNS UINT8 --coltype AMPLITUDE FLOAT64 \
        --coltype FWHM FLOAT64 --coltype RANGE FLOAT64 --pulsecols GPS_TIME,X_IDX,Y_IDX,Z_IDX

    # Printing info on Riegl file
    pylidar_info --input data.rxp
    