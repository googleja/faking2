import os


def rename_jpg_files(folder_path):
    """
    Renames all .jpg files in the specified folder to a sequential order (1.jpg, 2.jpg, ...).

    Args:
        folder_path (str): The path to the folder containing .jpg files.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Get all .jpg files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

    if not jpg_files:
        print("No .jpg files found in the folder.")
        return

    # Sort files to ensure consistent renaming order
    jpg_files.sort()

    # Rename files sequentially
    for idx, file_name in enumerate(jpg_files, start=1):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{idx}.jpg"
        new_path = os.path.join(folder_path, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"Renamed '{file_name}' to '{new_name}'")
        except Exception as e:
            print(f"Failed to rename '{file_name}': {e}")


if __name__ == "__main__":
    # Specify the folder containing .jpg files
    # folder_path = input("Enter the path to the folder containing .jpg files: ")
    folder_path = f"/home/jack/wvn/SurgicalDINO/data/negative"
    rename_jpg_files(folder_path)
