import cv2
import numpy as np
import os

def upscale_image(image_path, scale_factor, output_width=None, output_height=None):
    """
    Upscales an image while maintaining aspect ratio using bicubic interpolation.

    Args:
        image_path: Path to the input image file.
        scale_factor: The factor by which to scale dimensions (e.g., 2.0 for double size).
        output_width: (Optional) Desired output width in pixels.
        output_height: (Optional) Desired output height in pixels.

    Returns:
        The upscaled image as a NumPy array.
    """

    # Load the image
    img = cv2.imread(image_path)

    # Determine output dimensions based on aspect ratio
    if output_width is not None and output_height is not None:
        # Both width and height specified, ignore scale_factor
        new_width = output_width
        new_height = output_height
    elif output_width is not None:
        # Only width specified, calculate height to maintain aspect ratio
        new_width = output_width
        new_height = int(img.shape[0] * (output_width / img.shape[1]))
    elif output_height is not None:
        # Only height specified, calculate width to maintain aspect ratio
        new_height = output_height
        new_width = int(img.shape[1] * (output_height / img.shape[0]))
    else:
        # Use scale_factor for both dimensions
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)

    # Upscale the image using bicubic interpolation
    upscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return upscaled_img

# Example usage
image_path = "./ressources/set5/baby.png"
scale_factor = 2.0
output_folder = "./ressources/set5"  # Define the output folder

upscaled_image = upscale_image(image_path, scale_factor)

# Construct the full output file path
output_filename = "baby2x.png"
output_file_path = os.path.join(output_folder, output_filename)

# Save the upscaled image to the specified folder
cv2.imwrite(output_file_path, upscaled_image)