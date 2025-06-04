Run `code/privacy_masking_all_synthetic.ipynb`, with the preferred global variables to get output detection, masked, and depth images, in the respective output directory


Run `code/reorganize.py` to rename the output images correctly for feed into merge_pointclouds.py (if there is more than 1 view)


Run `code/merge_pointclouds.py` with the preferred variables in main to get 3d point cloud outputs under `[dataset]_frames`


For use of inpainting, stable diffusion, or gan models. Use the respective huggingface models within `privacy_masking_all_synthetic.ipynb`.
Or clone the `pixel2style2pixel` and `deepfill` library and use commented out imports in `code/privacy_masking_all_synthetic.ipynb`.