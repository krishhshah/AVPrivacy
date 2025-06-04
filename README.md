
Run code/privacy_masking_all_synthetic.ipynb, with the preferred global variables to get output detection, masked, and depth images, in the respective output directory


Run code/reorganize.py to rename the output images correctly for feed into merge_pointclouds.py (if there is more than 1 view)


Run code/merge_pointclouds.py with the preferred variables in main to get 3d point cloud outputs under [dataset]_frames


For use of inpainting, stable diffusion, or gan models. Use the respective huggingface models within privacy_masking_all_synthetic.ipynb.
Or clone the pixel2style2pixel and deepfill library and use commented out imports in code/privacy_masking_all_synthetic.ipynb.


# Real-Time Face Anonymization for Enhanced Privacy in AV/XR

## Introduction / Goal
Our pipeline’s task:
- Anonymize all human faces in AV’s sensor data (RGB-D) in real-time.
- Use our privatized data to train models ⇒ output must adhere to normal human shape/looks

Core Components
- Utilize synthetic data for development and testing.
- Efficiently identify people and localizing their head regions using RGB-D data.
- Apply a fast and irreversible technique to make faces non-recognizable.

## Dataset Generation
Synthetic Data > Real Data
- Avoids collecting real-world data with identifiable faces
- Control over scenarios, lighting, number of people, and camera parameters.
- Simple togenerating ground truth for segmentation and anonymization.

Tooling: CARLA Simulator
- Simple setup and usage compared to competing simulators
- We capture RGB-D video sequences from virtual cameras mounted on simulated vehicles.
- This data is the input for our segmentation and anonymization pipeline.

### Carla Simulator
- Open-source driving simulator for AV research
- We ran simulator on NVIDIA RTX 4090 and communicated via Python API
- Able to add various stationary sensors or attach to vehicles:
  - RGB Camera
  - Depth Camera
  - LiDAR Camera
  - Semantic Segmentation (from preset actor tag)
![RGB+SemSeg+Depth+GT](assets/trimmed_four_vids.gif)

## Problems with Existing Solutions
- To pinpoint people, need to use instance/object segmentation (typically uses U-net architecture)
- Powerful deep learning segmentors are expensive
  - They cannot be run in real-time (i.e., ≥ 30 frames per second)
- Object detection is faster but provides coarse-grained output → risk of overmasking

## Our Proposal
Use object detection supplemented with “depth” data (RGB-D):
- Depth provides an additional layer for segmentation
- All pixels belonging to the same object fall within a small range of depth values
- Use a fast object detector (instead of an instance segmentor) and the depth values within the bounding box

## Facial Anonymization
Make faces non-recognizable to humans while being fast and difficult to reverse
Explored methods:
- Generative adversarial networks (GANs)
- Inpainting
- Stable diffusion
- Simple gaussian blur/noise
- Image resampling
