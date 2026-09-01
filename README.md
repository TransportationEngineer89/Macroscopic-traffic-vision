# Macroscopic Traffic Vision Analyzer

A computer vision toolkit designed for automated macroscopic traffic data extraction. It utilizes YOLO object detection and OpenCV to analyze spot speeds, time and space headways, and traffic flow states from continuous video feeds.

## Features
*   **Object Tracking:** Utilizes ByteTrack integrated with YOLO for consistent, frame-to-frame vehicle ID tracking.
*   **Macroscopic Flow Calculations:** Automates the extraction of space and time headways using fundamental distance-over-time geometry across virtual trap lines.
*   **Custom Classification Thresholding:** Includes editable pixel-area boundaries to dynamically classify Light Commercial Vehicles (LCVs) based on bounding box dimensions.
*   **Flow State Categorization:** Automatically classifies the traffic stream into "Free Flow", "Stream Flow", or "Lead Vehicle" based on real-time space headway calculations.

## Scripts Overview

### 1. `Vehicle Composition.py`
Tracks and counts vehicles crossing predefined intersection lines. It features custom logic to classify Light Commercial Vehicles (LCVs) based on pixel area thresholds, dynamically redefining small trucks and large cars as LCVs upon crossing the designated lanes.

### 2. `Free Flow Speed.py`
Designed strictly for flow metric analysis, explicitly bypassing individual vehicle classification to optimize for raw volume and speed extraction. It establishes a 50m virtual speed trap to extract spot speed (km/h), time headway (s), and space headway (m).

### 3. `Space_and_Time_Headway.py`
A comprehensive extraction tool that combines the 50m speed trap methodology with vehicle classification and LCV bounding-box area evaluation. It provides a detailed dataset containing localized speed, headways, flow states, and vehicle class for deep-dive traffic flow analysis.

## Prerequisites & Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install opencv-python ultralytics pandas
```

### Hardware Acceleration (Recommended)
For optimal performance and real-time video processing, executing these scripts on a CUDA-enabled NVIDIA GPU (e.g., RTX 3050 or higher) is highly recommended. Ensure your NVIDIA drivers and CUDA toolkit are properly configured for your Windows 11 environment to allow PyTorch and YOLO to utilize the CUDA cores.

## Usage

1. Open the script you wish to run in your preferred IDE (e.g., VS Code).
2. Update the `video_path` variable to point to your local traffic video file.
3. If necessary, adjust the coordinate tuples for the entry/exit lines (`lane1_entry`, `lane1_exit`, etc.) to match the camera angle and resolution (1920x1080) of your specific footage.
4. Run the script. Press `q` to terminate the video window early. The output will automatically save to a `.csv` file in the same directory.

## Acknowledgements

This project utilizes the following open-source tools and libraries. We are grateful to their contributors:

*   **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics):** Used for robust object detection and tracking. Licensed under the AGPL-3.0 License.
*   **[OpenCV](https://github.com/opencv/opencv):** Used for video frame processing, coordinate mapping, and visual rendering. Licensed under the Apache License 2.0.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details. 

*Note: Because this project relies on Ultralytics YOLO (which is AGPL-3.0 licensed), this repository inherits the same strong copyleft license. Any modifications or derivative works must also be open-sourced under AGPL-3.0.*
