# Overview

The `Detector2DPlugin` has been refactored to seamlessly support multiple models (TransUNet, RITnet, and the legacy 2D C++), provide a toggle for visual comparison, and handle hardware-specific vertical flips for Eye 0. 

## Key Changes

### 1. `detector_base_plugin.py` Refactored
- Removed the hardcoded model dispatching branches.
- The base plugin now simply calls `self.detect(...)`, restoring the proper object-oriented hierarchy.

### 2. `detector_2d_plugin.py` Feature Implementation
- **UI Menu**:
  - `Active Model`: A dropdown selector to choose between `TransUNet`, `RITnet`, and `2D C++`. Changing this will dynamically route the incoming frames to the respective detector backend.
  - `Flip Vertically (Eye 0)`: A toggle switch allowing the user to correct the upside-down input commonly seen in the left eye camera (Eye 0).
  - `Flip Horizontally (Eye 0)`: A toggle switch added to correctly flip the right eye so it matches the left-eye-only training data distribution of the OpenEDS dataset.
  - `Show RITnet vs TransUNet`: A clean toggle to turn the `ComparisonVisualizer` window on and off.
- **RITnet Integration**:
  - Implemented `_detect_ritnet` which leverages the PyTorch models initialized during the plugin startup.
  - Added CPU Thread limiting (`torch.set_num_threads(1)`) to prevent PyTorch from hogging CPU resources and starving the USB camera `uvc_backend`, which previously caused "Corrupt JPEG" glitches and severe calibration error spikes.
- **TransUNet Integration (Replaces U-Mamba)**:
  - **Dynamic Loading**: Injects `~/PycharmProjects/transUnet` into `sys.path` to dynamically load the `VisionTransformer` (`R50-ViT-B_16`) model architecture and weights.
  - **Pre-processing (Domain Gap Fix)**: Applies Gamma Correction and CLAHE normalizations to standardize the IR camera brightness, followed by a conversion to `RGB 224x224` as expected by TransUNet.
  - **Horizontal & Vertical Flip Coordinate Math**: When a flip toggle is active, the image is physically flipped (`cv2.flip`) before being fed into the neural networks. After fitting the pupil ellipse, the coordinates (`cx`, `cy` and `angle_deg`) are inverted back to the original reference frame.

> [!TIP]
> Go to the Pupil Capture UI for **Eye 0**, enable **Flip Horizontally**, and set **Active Model** to **TransUNet**. The segmentation will now be robust against elliptical deformations!
