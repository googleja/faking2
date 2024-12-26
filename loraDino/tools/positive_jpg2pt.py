import cv2
import torch
import numpy as np
from torchvision import transforms


def jpg_to_pt(jpg_path, pt_path, device="cpu"):
    """
    Converts a JPG image to a tensor and saves it as a .pt file.

    Args:
        jpg_path (str): Path to the input .jpg file.
        pt_path (str): Path to save the output .pt file.
        device (str): Device to store the tensor (e.g., "cpu" or "cuda").

    Returns:
        None
    """
    # Read the image using OpenCV
    np_image = cv2.imread(jpg_path)
    if np_image is None:
        raise FileNotFoundError(f"Image file '{jpg_path}' not found.")

    # Convert BGR to RGB
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    transform = transforms.ToTensor()
    tensor = transform(np_image).to(device)

    # Save the tensor as a .pt file
    torch.save(tensor, pt_path)
    print(f"Saved tensor to {pt_path}")


if __name__ == "__main__":
    for i in range(1, 64):
        jpg_path = f"/home/jack/wvn/SurgicalDINO/data/positive/{i}.jpg"
        pt_path = f"/home/jack/wvn/SurgicalDINO/data/positive/{i}.pt"
        jpg_to_pt(jpg_path, pt_path, device="cuda")

    # jpg_path = f"/home/jack/wvn/SurgicalDINO/data/key/10.jpg"
    # pt_path = f"/home/jack/wvn/SurgicalDINO/data/key/_10.pt"
    # pt_orig = f"/home/jack/wvn/SurgicalDINO/data/key/10.pt"
    # # Convert JPG to PT
    # jpg_to_pt(jpg_path, pt_path, device="cuda")
    #
    # tensor1 = torch.load(pt_path)
    # tensor2 = torch.load(pt_orig)
    # print(sum(tensor1 - tensor2, 0))
    # print(torch.any(tensor1.cuda() != tensor2.cuda()))
