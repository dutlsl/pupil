# Overview

The `Detector2DPlugin` has been refactored to seamlessly support multiple models (U-Mamba, RITnet, and the legacy 2D C++), provide a toggle for visual comparison, and handle hardware-specific vertical flips for Eye 0. 

## Key Changes

### 1. `detector_base_plugin.py` Refactored
- Removed the hardcoded `hasattr(self, "detect_umamba")` branch.
- The base plugin now simply calls `self.detect(...)`, restoring the proper object-oriented hierarchy.

### 2. `detector_2d_plugin.py` Feature Implementation
- **UI Menu**:
  - `Active Model`: A dropdown selector to choose between `U-Mamba`, `RITnet`, and `2D C++`. Changing this will dynamically route the incoming frames to the respective detector backend.
  - `Flip Vertically (Eye 0)`: A toggle switch allowing the user to correct the upside-down input commonly seen in the left eye camera (Eye 0).
  - `Flip Horizontally (Eye 0)`: A toggle switch added to correctly flip the right eye so it matches the left-eye-only training data distribution of the OpenEDS dataset.
  - `Show RITnet vs U-Mamba`: A clean toggle to turn the `ComparisonVisualizer` window on and off.
- **RITnet Integration**:
  - Implemented `_detect_ritnet` which leverages the PyTorch models initialized during the plugin startup.
  - Ensured both models can be hot-swapped without restarting Pupil Capture.
- **Pre-processing and Normalization Updates (U-Mamba)**:
  - **Horizontal & Vertical Flip Coordinate Math**: When a flip toggle is active, the image is physically flipped (`cv2.flip`) before being fed into the neural networks. After fitting the pupil ellipse, the coordinates (`cx`, `cy` and `angle_deg`) are inverted back to the original reference frame.
  - **Letterbox Resizing**: Replaced standard stretching with letterbox resizing (padding 400x400 to 640x400) to preserve the original circular shape of the pupil, eliminating severe horizontal stretching distortion.
  - **CLAHE Normalization**: Integrated Gamma Correction + CLAHE into the U-Mamba pipeline (matching RITnet) to bridge the domain gap between Pupil Labs IR cameras and OpenEDS VR HMD images.

> [!TIP]
> Go to the Pupil Capture UI for **Eye 0**, enable **Flip Horizontally**, and set **Active Model** to **U-Mamba**. The segmentation will now be correctly interpreted as a left eye, bypassing the OpenEDS bias!
