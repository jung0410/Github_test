## Project Overview
This repository provides the implementation of a computer vision–based displacement
measurement framework proposed in our paper:

"Title of Your Paper" (Journal, Year). 나중에 쓰기

The code supports marker extraction, camera calibration,
and distortion correction for displacement measurement experiments.

```text
.
├── Demo
├──Algorithm
│  ├── A1_Marker_Detection.py
│  ├── A2_Camera_Calibration.py
│  ├── A3_Reprojection_Error_Calculation.py
│  ├── A4_Lens_Distortion_Correction.py
│  ├── A5_Homography_Correction.py
│  └── A6_Deformation_Measurement.py
├── Test_Method_Result
│   └── Arduino_Computer_Connection_Code
├── lib
│   ├── L1_Image_Conversion.py
│   ├── L2_Point_Detection_Conversion.py
│   ├── L3_Zhang_Camera_Calibration.py
│   ├── L4_Pipeline_Utilities.py
│   └── L5_Visualization_Utilities.py
└── main.py
```

### Algortihm
- **A1_Marker_detection.py**: Marker detection algorithm (Algorithm 1).
- **A2_Camera_calibration.py**: Camera calibration using Zhang’s method (Algorithm 2).
- **A3_Reprojection_Error_calculation.py**: Reprojection error evaluation (Algorithm 3).
- **A4_Lens_distortion_correction.py**: Lens distortion correction (Algorithm 4).
- **A5_Homography_correction.py**: Perspective correction via homography (Algorithm 5).
- **A6_Deformation_measurement.py**: Final displacement and deformation measurement.

  
### lib
This directory contains core utility modules used across the processing pipeline:
- **L1_Image_Conversion.py**: Image generation and format conversion utilities.
- **L2_Point_Detection.py**: Feature and marker point detection functions.
- **L3_Zhang_Camera_Calibration.py**: Camera calibration based on Zhang’s method.
- **L4_Pipeline_Utilities.py**: Shared utilities for pipeline execution in `main.py`.
- **L5_Visualization_Utilities.py**: Visualization tools for points and images.

**main.py**: Executes the complete CV-based displacement measurement pipeline by
  sequentially applying Algorithms 1–5 in a single run.
