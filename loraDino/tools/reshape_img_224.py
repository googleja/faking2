from PIL import Image
import os


"""主要用于将不是224x224的图片从右侧裁减一块，转为224x224，主要是处理RUGD数据集"""


def resize_image_to_square(input_path, output_path, target_size=224):
    """
    将图片裁剪为正方形并调整为指定大小。

    Args:
        input_path (str): 输入图片路径。
        output_path (str): 输出图片路径。
        target_size (int): 调整后的图片大小，默认为 224。

    Returns:
        None
    """
    # 打开图片
    image = Image.open(input_path)

    # 获取原始图片的尺寸
    width, height = image.size

    # 计算裁剪区域，使其裁减右侧，变为正方形
    if width > height:
        # left = (width - height) // 2
        left = 0
        top = 0
        right = left + height
        bottom = height
    else:
        top = (height - width) // 2
        left = 0
        right = width
        bottom = top + width

    # 裁剪为正方形
    image = image.crop((left, top, right, bottom))

    # 调整为目标大小
    # image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    image = image.resize((target_size, target_size), Image.NEAREST)

    # 保存图片
    image.save(output_path)


def process_folder(input_folder, output_folder, target_size=224):
    """
    对文件夹中的所有图片执行裁剪和调整操作。

    Args:
        input_folder (str): 输入文件夹路径。
        output_folder (str): 输出文件夹路径。
        target_size (int): 调整后的图片大小，默认为 224。

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 确保只处理图片文件
        if os.path.isfile(input_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                resize_image_to_square(input_path, output_path, target_size)
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")


if __name__ == "__main__":
    input_folder = "/home/jack/wvn/SurgicalDINO/data1219/annotation"  # 替换为你的输入文件夹路径
    output_folder = "/home/jack/wvn/SurgicalDINO/data1219/annotation"  # 替换为你的输出文件夹路径
    process_folder(input_folder, output_folder)
