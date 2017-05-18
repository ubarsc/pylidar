SPD V4 Format Description
=========================

Attributes on a SPD V4 file
---------------------------

Fields that were part of the 'HEADER' dataset in V3 have now been moved into 
the attributes on the file. Other optional fields can be created but the ones
below are understood by PyLidar.

+------------------------------------+---------+-------+-----------+------------------------------------+
| Name                               | Type    | Units | Essential | Description                        |
+====================================+=========+=======+===========+====================================+
| AZIMUTH_MAX                        | float64 | deg   | No        | Maximum pulse azimith in this file |
+------------------------------------+---------+-------+-----------+------------------------------------+
| AZIMUTH_MIN                        | float64 | deg   | No        | Minimum pulse azimith in this file |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BANDWIDTHS                         | float32 | ns    | No        | Widths of the band(s) used         |
|                                    | array   |       |           |                                    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BIN_SIZE                           | float32 | m/deg | No        | Size of the bins used for spatial  |
|                                    |         |       |           | index. See INDEX_TYPE              |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BLOCK_SIZE_POINT                   | uint16  |       | Yes       | HDF5 chunk size for point arrays   |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BLOCK_SIZE_PULSE                   | uint16  |       | Yes       | HDF5 chunk size for pulse arrays   |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BLOCK_SIZE_WAVEFORM                | uint16  |       | Yes       | HDF5 chunk size for waveform array |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BLOCK_SIZE_RECEIVED                | uint16  |       | Yes       | HDF5 chunk size for recv array     |
+------------------------------------+---------+-------+-----------+------------------------------------+
| BLOCK_SIZE_TRANSMITTED             | uint16  |       | Yes       | HDF5 chunk size for trans array    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| CAPTURE_DATETIME                   | str     |       | Yes       | UTC ISO String of capture date/time|
+------------------------------------+---------+-------+-----------+------------------------------------+
| CREATION_DATETIME                  | str     |       | Yes       | UTC ISO String of file creation    |
|                                    |         |       |           | date/time                          |
+------------------------------------+---------+-------+-----------+------------------------------------+
| FIELD_OF_VIEW                      | float32 | deg   | No        | Field of view of the sensor        |
+------------------------------------+---------+-------+-----------+------------------------------------+
| FILE_TYPE                          | uint16  |       | Yes       | 0=No index; 1=Sequential;          |
|                                    |         |       |           | 2=Non-sequential                   |
+------------------------------------+---------+-------+-----------+------------------------------------+
| GENERATING_SOFTWARE                | str     |       | Yes       | Name and version of software used  |
|                                    |         |       |           | to create this file                |
+------------------------------------+---------+-------+-----------+------------------------------------+
| INDEX_TYPE                         | uint16  |       | Yes       | Type of spatial index              |
|                                    |         |       |           | 0=None; 1=Cartesian; 2=Spherical;  |
|                                    |         |       |           | 3=Cylindrical; 4=Polar; 5=Scan     |
+------------------------------------+---------+-------+-----------+------------------------------------+
| INDEX_TLX                          | float64 |       | No        | Top left coord of spatial index    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| INDEX_TLY                          | float64 |       | No        | Top left coord of spatial index    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| NUMBER_BINS_X                      | uint32  |       | No        | number of bins in spatial index    |
|                                    |         |       |           | in x coordinate                    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| NUMBER_BINS_Y                      | uint32  |       | No        | number of bins in spatial index    |
|                                    |         |       |           | in y coordinate                    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| NUMBER_OF_POINTS                   | uint64  |       | Yes       | Number of points in the file       | 
+------------------------------------+---------+-------+-----------+------------------------------------+
| NUMBER_OF_PULSES                   | uint64  |       | Yes       | Number of pulses in the file       | 
+------------------------------------+---------+-------+-----------+------------------------------------+
| NUMBER_OF_WAVEFORMS                | uint64  |       | Yes       | Number of waveforms (transmitted   |
|                                    |         |       |           | and received) in the file          |
+------------------------------------+---------+-------+-----------+------------------------------------+
| NUM_OF_WAVELENGTHS                 | uint16  |       | No        | Number of different wavelengths in |
|                                    |         |       |           | the file                           |
+------------------------------------+---------+-------+-----------+------------------------------------+
| POINT_DENSITY                      | float32 | p/m2  | No        | Number of points per square meter  |
|                                    |         |       |           | on average                         |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_ALONG_TRACK_SPACING          | float32 | m     | No        | gap between pulses in direction of |
|                                    |         |       |           | instrument movement                |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_ACROSS_TRACK_SPACING         | float32 | m     | No        | gap between pulses in direction    |
|                                    |         |       |           | orthogonal to instrument movement  |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_ANGULAR_SPACING_SCANLINE     | float32 | deg   | No        | Average pulse spacing between      |
|                                    |         |       |           | scanlines                          |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_ANGULAR_SPACING_SCANLINE_IDX | float32 | deg   | No        | Average pulse spacing between      |
|                                    |         |       |           | pulses along a scanlines           |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_DENSITY                      | float32 | p/m2  | No        | Number of pulses per square meter  |
|                                    |         |       |           | on average                         |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_ENERGY                       | float32 | J     | No        | Amount of energy in each           |
|                                    |         |       |           | transmitted pulse                  |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_FOOTPRINT                    | float32 | m2    | No        | Size of each pulse on the ground   |
+------------------------------------+---------+-------+-----------+------------------------------------+
| PULSE_INDEX_METHOD                 | uint16  |       | Yes       | 0=FIRST_RETURN; 1=LAST_RETURN;     |
|                                    |         |       |           | 2=START_WAVEFORM; 3=END_WAVEFORM;  |
|                                    |         |       |           | 4=ORIGIN; 5=MAX_INTENSITY          |
+------------------------------------+---------+-------+-----------+------------------------------------+
| RANGE_MAX                          | float32 | m     | No        | Maximum range of returns           |
+------------------------------------+---------+-------+-----------+------------------------------------+
| RANGE_MIN                          | float32 | m     | No        | Minimum range of returns           |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SCANLINE_IDX_MAX                   | uint32  |       | No        | Maximum scanline index number      |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SCANLINE_IDX_MIN                   | uint32  |       | No        | Minimum scanline index number      |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SCANLINE_MAX                       | uint16  |       | No        | Maximum scanline number            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SCANLINE_MIN                       | uint16  |       | No        | Minimum scanline number            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_APERTURE_SIZE               | float32 | m2    | No        | Size of aperture                   |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_BEAM_DIVERGENCE             | float32 | mrad  | No        | Laser beam divergence              |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_HEIGHT                      | float32 | m     | No        | Height of sensor                   |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_MAX_SCAN_ANGLE              | float32 | deg   | No        | Maximum scan angle of sensor       |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_PULSE_REPETITION_FREQ       | float32 | kHz   | No        | How often pulses are sent out      |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_SCAN_RATE                   | float32 | Hz    | No        | How often a new scan line is done  |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_SPEED                       | float32 | km/h  | No        | How fast sensor is moving forward  |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_TEMPORAL_BIN_SPACING        | float64 | ns    | No        | Waveform bin size                  |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SENSOR_BEAM_EXIT_DIAMETER          | float32 | m     | No        | Laser beam diameter at sensor exit |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SPATIAL_REFERENCE                  | str     |       | Yes       | Well known text (WKT) describing   |
|                                    |         |       |           | coordinate system of file          |
+------------------------------------+---------+-------+-----------+------------------------------------+
| SYSTEM_IDENTIFIER                  | str     |       | No        | How the file was generated         |
+------------------------------------+---------+-------+-----------+------------------------------------+
| USER_META_DATA                     | str     |       | No        | User can put there own metadata    |
|                                    |         |       |           | here as a JSON or XML string       |
+------------------------------------+---------+-------+-----------+------------------------------------+
| VERSION_SPD                        | uint8   |       | Yes       | An array of version information    |
|                                    | array   |       |           | for this file format               |
+------------------------------------+---------+-------+-----------+------------------------------------+
| VERSION_DATA                       | uint8   |       | Yes       | An array of version information    |
|                                    | array   |       |           | for the file data                  |
+------------------------------------+---------+-------+-----------+------------------------------------+
| WAVEFORM_BIT_RES                   | uint16  |       | No        | Nominal waveform radiometric       |
|                                    |         |       |           | resolution                         |
+------------------------------------+---------+-------+-----------+------------------------------------+
| WAVELENGTHS                        | float32 | nm    | No        | Wavelengths used in the file       |
|                                    | array   |       |           |                                    |
+------------------------------------+---------+-------+-----------+------------------------------------+
| X_MAX                              | float64 | m     | No        | maximum X coord in file            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| X_MIN                              | float64 | m     | No        | minimum X coord in file            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| Y_MAX                              | float64 | m     | No        | maximum Y coord in file            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| Y_MIN                              | float64 | m     | No        | minimum X coord in file            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| Z_MAX                              | float64 | m     | No        | maximum Z coord in file            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| Z_MIN                              | float64 | m     | No        | minimum Z coord in file            |
+------------------------------------+---------+-------+-----------+------------------------------------+
| HEIGHT_MIN                         | float32 | m     | No        | minimum height in file             |
+------------------------------------+---------+-------+-----------+------------------------------------+
| HEIGHT_MAX                         | float32 | m     | No        | maximum height in file             |
+------------------------------------+---------+-------+-----------+------------------------------------+
| ZENITH_MAX                         | float64 | m     | No        | maximum zenith in file             |
+------------------------------------+---------+-------+-----------+------------------------------------+
| ZENITH_MIN                         | float64 | m     | No        | minimum zenith in file             |
+------------------------------------+---------+-------+-----------+------------------------------------+
| RGB_FIELD                          | str     |       | No        | List of 3 Point columns to use for | 
|                                    |         |       |           | visualisation                      |
+------------------------------------+---------+-------+-----------+------------------------------------+

Pulse data
----------

Pulse data live under the DATA/PULSES group. Each column is a separate dataset. These are the fields that
PyLidar recognises. Other optional fields can be created. If a field has 'GAIN' and 'OFFSET' attributes
these will be applied to data transparently. Request unscaled versions by appending '_U' to field name.

Fields marked with Scaling = Yes must have these attributes.

+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| Name                       | Type    | Units | Essential | Scaling | Description                                  |
+============================+=========+=======+===========+=========+==============================================+
| PULSE_ID                   | uint64  |       | Yes       | No      | A unique number identifying this pulse       |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| TIMESTAMP                  | uint64  | ns    | No        | No      | GPS time or system time                      |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| NUMBER_OF_RETURNS          | uint8   |       | Yes       | No      | Number of points for this pulse              |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| AZIMUTH                    | uint32  | rad   | No        | Yes     | Azimuth of this pulse from true north        |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| ZENITH                     | uint32  | rad   | No        | Yes     | Zenith of this pulse                         |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| SOURCE_ID                  | uint16  |       | No        | No      | Pulse source (typically a flightline ID      |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| PULSE_WAVELENGTH_IDX       | uint8   |       | No        | No      | Index into WAVELENGTHS file attribute        |
|                            |         |       |           |         | that this pulse uses                         |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| NUMBER_OF_WAVEFORM_SAMPLES | uint8   |       | No        | No      | Some instruments (e.g. RIEGL) only record    |
|                            |         |       |           |         | parts of the full waveform, and sometimes the|
|                            |         |       |           |         | same part in multiple channels               |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| WFM_START_IDX              | uint64  |       | No        | No      | Offset into WAVEFORM records                 |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| PTS_START_IDX              | uint64  |       | Yes       | No      | Offset into POINTS records                   |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| SCANLINE                   | uint32  |       | No        | No      | Scanline number                              |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| SCANLINE_IDX               | uint16  |       | No        | No      | Pulse number within a scanline               |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| X_IDX                      | uint32  |       | No        | Yes     | X coord to use to spatially index this pulse |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| Y_IDX                      | uint32  |       | No        | Yes     | Y coord to use to spatially index this pulse |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| X_ORIGIN                   | uint32  | m     | No        | Yes     | X location of pulse emission                 |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| Y_ORIGIN                   | uint32  | m     | No        | Yes     | Y location of pulse emission                 |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| Z_ORIGIN                   | uint32  | m     | No        | Yes     | Z location of pulse emission                 |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| H_ORIGIN                   | uint32  | m     | No        | Yes     | Height of pulse emission                     |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| PULSE_FLAGS                | uint8   |       | No        | No      | 1=IGNORE,2=OVERLAP,4=SCANLINE_DIRECTION,     |
|                            |         |       |           |         | 8=SCANLINE_EDGE                              |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| AMPLITUDE_PULSE            | uint16  |       | No        | Yes     | Amplitude of the emitted pulse               |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| WIDTH_PULSE                | uint16  | ns    | No        | Yes     | Width (FWHM) of the emitted pulse            |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| SCAN_ANGLE_RANK            | int16   | deg   | No        | No      | In LAS specification and defined differently |
|                            |         |       |           |         | to zenith angle                              |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+
| PRISM_FACET                | uint8   |       | No        | No      | For RIEGL ALS and TLS data that use a        |
|                            |         |       |           |         | rotating prism instead of a scanning mirror  |
+----------------------------+---------+-------+-----------+---------+----------------------------------------------+

Point Data
----------

Point data live under the DATA/POINTS group. Each column is a separate dataset. These are the fields that
PyLidar recognises. Other optional fields can be created. If a field has 'GAIN' and 'OFFSET' attributes
these will be applied to data transparently. Request unscaled versions by appending '_U' to field name.

Fields marked with Scaling = Yes must have these attributes.

+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| Name                 | Type    | Units | Essential | Scaling | Description                                  |
+======================+=========+=======+===========+=========+==============================================+
| RANGE                | uint32  | m     | No        | Yes     | Return range                                 |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| RETURN_NUMBER        | uint8   |       | Yes       | No      | Return number. The base value is 1 not 0     |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| X                    | uint32  | m     | Yes       | Yes     | The X coord of this point                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| Y                    | uint32  | m     | Yes       | Yes     | The Y coord of this point                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| Z                    | uint32  | m     | Yes       | Yes     | The Z coord of this point                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| HEIGHT               | uint16  | m     | Yes       | Yes     | The height of this point                     |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| CLASSIFICATION       | uint8   |       | Yes       | No      | Some user defined classification of the point|
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| POINT_FLAGS          | uint8   |       | No        | No      | 1=IGNORE,2=OVERLAP,4=SYNTHETIC,8=KEY_POINT,  |
|                      |         |       |           |         | 16=WAVEFORM                                  |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| INTENSITY            | uint16  |       | No        | Yes     | Uncalibrated intensity                       |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| AMPLITUDE_RETURN     | uint16  |       | No        | Yes     | Amplitude of the return                      |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| WIDTH_RETURN         | uint16  | ns    | No        | Yes     | Width (FWHM) of the return                   |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| RED                  | uint16  |       | No        | No      | For display purposes. See RGB_FIELD file     |
|                      |         |       |           |         | attribute                                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| GREEN                | uint16  |       | No        | No      | For display purposes. See RGB_FIELD file     |
|                      |         |       |           |         | attribute                                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| BLUE                 | uint16  |       | No        | No      | For display purposes. See RGB_FIELD file     |
|                      |         |       |           |         | attribute                                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| NIR                  | uint16  |       | No        | No      | For display purposes. See RGB_FIELD file     |
|                      |         |       |           |         | attribute                                    |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| RHO_APP              | uint32  |       | No        | Yes     | Apparent reflectance                         |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| DEVIATION            | uint16  |       | No        | No      | Return deviation (defined by RIEGL)          |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| ECHO_TYPE            | uint16  |       | No        | No      | Useful for old datasets where first/last     |
|                      |         |       |           |         | returns are provided independently, i.e.     |
|                      |         |       |           |         | pulse structure is otherwise unknown         |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+
| POINT_WAVELENGTH_IDX | uint8   |       | No        | No      | Only use as point attribute if instrument is |
|                      |         |       |           |         | a white laser so only a single pulse for     |
|                      |         |       |           |         | multiple wavelength readings                 |
+----------------------+---------+-------+-----------+---------+----------------------------------------------+

Waveform Data
-------------

Waveform data live under the DATA/WAVEFORMS group. Each column is a separate dataset. These are the fields that
PyLidar recognises. Other optional fields can be created. If a field has 'GAIN' and 'OFFSET' attributes
these will be applied to data transparently. Request unscaled versions by appending '_U' to field name.
The actual transmitted and received data is in the DATA/TRANSMITTED and DATA/RECEIVED datasets respectively.

Fields marked with Scaling = Yes must have these attributes.

+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| Name                                | Type    | Units | Essential | Scaling | Description                                  |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| NUMBER_OF_WAVEFORM_RECEIVED_BINS    | uint16  |       | Yes       | No      | Number of received bins for this waveform    |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| NUMBER_OF_WAVEFORM_TRANSMITTED_BINS | uint16  |       | Yes       | No      | Number of transmitted bins for this waveform |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| RANGE_TO_WAVEFORM_START             | uint32  | m     | Yes       | Yes     | distance to the start of this waveform from  |
|                                     |         |       |           |         | the instrument                               |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| RECEIVED_START_IDX                  | uint64  |       | Yes       | No      | Index into the RECEIVED dataset              |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| TRANSMITTED_START_IDX               | uint64  |       | Yes       | No      | Index into the TRANSMITTED dataset           |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| CHANNEL                             | uint8   |       | No        | No      | Channel number (e.g. for high/low gain)      |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| WAVEFORM_FLAGS                      | uint8   |       | No        | No      | 1=IGNORE,2=SATURATION_FIXED,4=BASELINE_FIXED |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| WFM_WAVELENGTH_IDX                  | uint8   |       | Yes       | No      | Index of this waveform                       |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| RECEIVE_WAVE_GAIN                   | float32 |       | Yes       | No      | Gain for the RECEIVED data                   |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| RECEIVE_WAVE_OFFSET                 | float32 |       | Yes       | No      | Offset for the RECEIVED data                 |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| TRANS_WAVE_GAIN                     | float32 |       | Yes       | No      | Gain for the TRANSMITTED data                |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+
| TRANS_WAVE_OFFSET                   | float32 |       | Yes       | No      | Offset for the TRANSMITTED data              |
+-------------------------------------+---------+-------+-----------+---------+----------------------------------------------+


