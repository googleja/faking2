import os
import torch
from torchvision import transforms
from PIL import Image


def process_and_save_files(input_folder, output_folder):
    """
    Applies a transform to .jpg and .pt files in the input folder and saves the results to the output folder.
    The transformations for .jpg and .pt files are synchronized to ensure consistency.

    Args:
        input_folder (str): Path to the folder containing .jpg and .pt files.
        output_folder (str): Path to the folder where transformed files will be saved.

    Returns:
        None
    """
    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define a common transform
    common_transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])

    # Process files
    for file_name in os.listdir(input_folder):
        base_name, ext = os.path.splitext(file_name)

        if ext.lower() == '.jpg':
            try:
                # Apply transform to the image
                jpg_path = os.path.join(input_folder, file_name)
                image = Image.open(jpg_path).convert('RGB')
                transformed_image = common_transform(image)

                # Save transformed image
                image_output_path = os.path.join(output_folder, file_name)
                transformed_image_pil = transforms.ToPILImage()(transformed_image)
                transformed_image_pil.save(image_output_path)
                print(f"Processed and saved image: {file_name}")

            except Exception as e:
                print(f"Failed to process image '{file_name}': {e}")


if __name__ == "__main__":
    # Specify input and output folders
    input_folder = f"/home/jack/wvn/SurgicalDINO/data/key"
    output_folder = f"/home/jack/wvn/SurgicalDINO/data/positive"
    process_and_save_files(input_folder, output_folder)
