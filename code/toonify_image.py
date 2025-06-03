import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
from argparse import Namespace
from typing import Union
from models.psp import pSp
from utils.common import tensor2im

def load_toonify_model(psp_toonify_model_path: str = "pretrained_models/psp_toonify.pt"):
    """
    Loads the pre-trained pSp Toonify model. This function should be called once.

    Args:
        psp_toonify_model_path (str): Path to the pre-trained pSp Toonify encoder model (.pt file).
                                      Defaults to 'pretrained_models/psp_toonify.pt'.

    Returns:
        models.psp.pSp: The loaded pSp network object.
    """
    
    if not os.path.exists(psp_toonify_model_path):
        raise FileNotFoundError(
            f"pSp Toonify model not found: {psp_toonify_model_path}. "
            "Please ensure you've downloaded 'psp_toonify.pt' into your 'pretrained_models' folder "
            "and that the path is correct."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load the checkpoint. map_location='cpu' ensures it loads even if on a CPU machine,
        # then we explicitly move it to the correct device later.
        # Added weights_only=False to address the WeightsUnpickler error.
        ckpt = torch.load(psp_toonify_model_path, map_location='cpu', weights_only=False)
        
        # The model's configuration (opts) is stored within the checkpoint.
        opts = ckpt['opts']

        if isinstance(opts, dict):
            opts = Namespace(**opts)
        # If it's already a Namespace, it remains as is.

        # Manually add output_size if it's missing from the loaded options.
        # This model generates 1024x1024 images.
        if not hasattr(opts, 'output_size'):
            opts.output_size = 1024
            print(f"Injected missing 'output_size' attribute into model options: {opts.output_size}")
        
        # Ensure compatibility with different pSp versions.
        if not hasattr(opts, 'learn_in_w'):
            opts.learn_in_w = False
        
        # Set the checkpoint path in opts; this is used by pSp's internal loading logic.
        opts.checkpoint_path = psp_toonify_model_path 

        # Initialize the pSp network with the loaded options.
        net = pSp(opts)
        net.eval()  # Set the network to evaluation mode (disables dropout, batchnorm updates, etc.)
        net.to(device) # Move the model to the chosen device (GPU or CPU)
        print(f"pSp Toonify model loaded successfully from: {psp_toonify_model_path}")
        return net
    except Exception as e:
        print(f"Error loading pSp Toonify model: {e}")
        raise # Re-raise the exception after printing, so the calling code knows it failed

def toonify_image_with_stylegan(
    input_image: Image.Image, # Now only accepts PIL Image.Image
    loaded_model: pSp # Accepts the pre-loaded model
) -> Image.Image:
    """
    Transforms a single input PIL Image object into a cartoon style using a pre-loaded
    pSp encoder specifically trained for Toonify. The entire image will be processed,
    without explicit face detection. The transformed image is returned as a PIL Image object.

    Args:
        input_image (PIL.Image.Image): The input PIL Image object.
        loaded_model (models.psp.pSp): The pre-loaded pSp network object obtained from load_toonify_model().

    Returns:
        PIL.Image.Image: The cartoonified PIL Image object.

    Raises:
        ValueError: If input_image is not a PIL Image.
        Exception: For other potential issues during inference.
    """
    # --- Input Validation and Image Loading ---
    if not isinstance(input_image, Image.Image):
        raise ValueError("Input image must be a PIL Image object.")
    
    original_image = input_image.convert("RGB") # Ensure it's in RGB format
    input_identifier = "PIL Image object" # For printing purposes

    # Determine the device to use (GPU if available, otherwise CPU)
    # The loaded_model should already be on the correct device, but we keep this for consistency
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. Preprocess the input image ---
    # Define the transformation pipeline for the input image.
    # The pSp encoder for Toonify is typically trained to expect 256x256 input.
    # Images are normalized to a range of [-1, 1].
    transform_input = transforms.Compose([
        transforms.Resize((256, 256)), # Resize image to the encoder's expected input size
        transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize pixel values
    ])
    
    # Apply transformations and add a batch dimension (unsqueeze(0))
    # Move the tensor to the chosen device and ensure float type.
    input_tensor = transform_input(original_image).unsqueeze(0).to(device).float()

    print(f"Input image ({input_identifier}) preprocessed.")

    # --- 3. Perform cartoonification (inversion and generation) ---
    with torch.no_grad(): # Disable gradient calculations for inference, saves memory and speeds up.
        # The pSp network directly performs the encoding and then generation.
        # 'return_latents=True' would return the latent codes if you needed them for further editing.
        # 'resize=False' means the output will be the native resolution of the StyleGAN (1024x1024 for this model).
        transformed_image_tensor, _ = loaded_model(input_tensor, return_latents=True, resize=False)

    # --- 4. Convert the output tensor to a PIL Image ---
    # tensor2im is a pSp utility that denormalizes the tensor from [-1, 1] to [0, 255]
    # and converts it to a NumPy array, then to a PIL Image.
    transformed_pil_image = tensor2im(transformed_image_tensor[0]) # Take the first image from the batch

    print("Cartoonified image generated in memory.")

    return transformed_pil_image