import os
from .utils.jn import run
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import re
import csv

from project_path import PROJECT_PATH  # dynamic project root
from .utils.all_devices import get_jnr_device

def JERSEY_NUMBER_RECOGNITION(query=None, material=[]):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(get_jnr_device())
    else:
        device = torch.device("cpu")
    cudnn.benchmark = True

    material_paths = []
    if material:
        for entry in material:  # process all material entries
            # Convert relative path to absolute path
            if not os.path.isabs(entry):
                # If path starts with materials/, prepend challenge/test/ path
                if entry.startswith("materials/"):
                    entry = os.path.join(PROJECT_PATH, "challenge/test", entry)
                else:
                    entry = os.path.join(PROJECT_PATH, entry)
            
            if os.path.isdir(entry):
                # If it's a directory, collect all .jpg files
                for root, _, files in os.walk(entry):
                    for file in files:
                        if file.lower().endswith(".jpg"):
                            material_paths.append(os.path.join(root, file))
            elif os.path.isfile(entry):
                # If it's a file, add it to the list
                material_paths.append(entry)

    def _sort_key(path: str) -> int:
        match = re.search(r"(\d+)\.jpg", os.path.basename(path))
        return int(match.group(1)) if match else 10 ** 9

    material_paths.sort(key=_sort_key)

    image_list = []
    for path in material_paths:
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            continue
        image_list.append(img)

    if not image_list:
        error_msg = f"No valid jersey images were provided to the Number Recognition tool.\n"
        error_msg += f"Tried to load {len(material_paths)} path(s):\n"
        for path in material_paths[:5]:  # show only the first 5
            error_msg += f"  - {path}\n"
        raise ValueError(error_msg)

    model_path = f"{PROJECT_PATH}/pipeline/toolbox/utils/legibility_resnet34_soccer_20240215.pth" # Replace with your actual model path
    qwen_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    ans, result = run(device, image_list, model_path, qwen_path, threshold=0.5)
    ans = -1 if ans == None else ans
    ans = f"The jersey number in the pictures is {ans}."
    return ans
