
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
  - Semantic Segmentation (from preset actor tag) --> Ground Truth

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
Make faces non-recognizable to humans while being fast and difficult to reverse.

Explored methods:
- Generative adversarial networks (GANs)
- Inpainting
- Stable diffusion
- Simple Transformations
- Image resampling

### Generative Adversarial Networks (GANs)

A competition between a **generator (G)** and **discriminator (D)**:
- G generates synthetic images based on random noise latent vector
- D distinguishes between real data (training set) and fake data produced by G
- As iterations continue, G improves at **falsifying images** and D improves at **detecting fakes**

→ Over time, G produces high-quality synthetic images that mimics the training set and D can no longer distinguish between.

#### GANs: DeepPrivacyV2
*Goal*: Create realistic anonymized images of entire human faces/figures
- Uses the Flickr Diverse Humans (FDH) dataset (1.5M images)
- Ensemble of detectors (DSFD, CSE, Mask R-CNN) for robust facial detection
- 3 generators: CSE-guided full-body generator, unconditional full-body generator, face generator
- Removes areas to be anonymized → generators complete the missing regions. 
  - Generator never directly observes the privacy-sensitive information
- Utilizes recursive stitching of pasting anonymized individuals into the original image

*Pros*:
- Faces look more realistic

*Cons*: 
- Output 9fps is too slow for real-time video footage
- Recursive stitching causes increasingly slower runtime with more figures are on the screen

#### GANs: StyleGAN
*Goal*: Generate photorealistic and controllable synthetic images
- Map random noise vector (X) to an intermediate latent space (Y)
- Transform Y into “styles” via transformations
- Styles control various visual features (i.e., smile, hair, etc)

*Pros*:
- Fast, generates image in one pass (non-recursive)
- Can use model: `Pixel2style2pixel`

*Cons*:
- Pretrained models not applicable to our use case
- Attempted application resulted in unusable images

### Inpainting
#### DeepFill
*Goal*: Reconstruct the missing parts of an image, and make the result indistinguishable from the original

<p></p>
Two-Stage Coarse-to-Fine network: 

1. Output a coarse inpainted image with an encoder-decoder with dilated convolutions. 
2. Enhance the output of the coarse inpainted image using another encoder-decoder. 
    1. This component uses a contextual attention layer to find consist textures to fill the hole. 

Finally, use GANs (GAN loss) to ensure final output is realistic

*Pros*: Easy to set up

*Cons*: 
- Slower runtime, but sufficient given our chunking optimization
- Completely removes the head because inpainting uses the background

### Stable Diffusion
*Goal*: Generate synthetic images by learning to reverse the process of turning image into noise, enabling it to turn noise into an image
- Forward: iteratively add small amounts of gaussian noise
- Backward: train a U-Net network to predict the noise that was added in the forward step
- Image generation: start with random noise, iteratively apply trained network, output a “denoised” image

*Pros*: 
- Diffusion models are not known to be quick
- Can use model: `runwayml/stable-diffusion-inpainting`

*Cons*: 
- Poor results and unacceptable performance

### Simple Transformations
#### Gaussian blur, matrix transformations, image smoothing

*Pros*: Extremely fast and simple compared to complex models

*Cons*: 
- Easily reversible if attacker knows the privacy preservation method.
- Not individually sufficient for a robust anonymization pipeline.

### Image Resampling

*Pros*:
- Downsampling the images and upsampling → guarantees information loss
- Anonymized face to but computer can still detect a “face” without a loss of privacy
- Simple and fast
- Very difficult to reverse

## Chosen Anonymization Technique

**Ensemble of lightweight methods --> Simple ≠ bad**

- Gaussian noise
  - Initial obfuscation
- Matrix transformations
  - Apply random transformations to distort features
- Image smoothing
  - Further blurring of finer details
- Image resampling
  - Final step to introduce information loss all around

## Results
20 FPS & N=5 Chunking
|![](assets/front_view.gif)<br>Front view|![](assets/side_view.gif)<br>Side view|
|:-:|:-:|

