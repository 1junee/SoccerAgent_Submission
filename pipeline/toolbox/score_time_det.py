import easyocr, sys
import re
import os
import time
from urllib.error import URLError
import torch
from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline/toolbox")
os.environ.setdefault('VLM_ENABLE_THINKING', '1')
from vlm import VLM_QWEN as VLM

_EASYOCR_READER = None
_EASYOCR_USE_GPU = None


def _get_easyocr_reader(use_gpu: bool):
    global _EASYOCR_READER, _EASYOCR_USE_GPU
    if _EASYOCR_READER is not None and _EASYOCR_USE_GPU == use_gpu:
        return _EASYOCR_READER
    reader = easyocr.Reader(['en'], download_enabled=True, gpu=use_gpu)
    _EASYOCR_READER = reader
    _EASYOCR_USE_GPU = use_gpu
    return reader

def extract_timestamp(image, max_retries=3):

    reader = None
    use_gpu_env = os.getenv("EASYOCR_USE_GPU")
    if use_gpu_env is None:
        use_gpu = False
    else:
        use_gpu = use_gpu_env == "1"
    for attempt in range(max_retries):
        try:
            reader = _get_easyocr_reader(use_gpu)
            break
        except URLError as e:
            print(f"Download failed easyocr model, Attempt: {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return "Cannot download easyocr model."
            time.sleep(5)  
        except Exception as e:
            if use_gpu:
                print(f"easyocr GPU init failed, retrying on CPU. Error: {e}")
                use_gpu = False
            else:
                if attempt == max_retries - 1:
                    return f"Failed to initialize easyocr reader: {e}"
                time.sleep(2)
            continue
    
    try:
        results = reader.readtext(image)
        
        time_pattern = re.compile(r'(\d{1,2})[:.-](\d{2})')
        
        for (bbox, text, confidence) in results:
            match = time_pattern.search(text)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                
                if 0 <= minutes <= 90 and 0 <= seconds < 60:
                    return f"The timestamp detected by easy easyocr is {minutes} minutes {seconds} seconds."
        
        return "Cannot find timestamp via easyocr."
    
    except Exception as e:
        return f"Failed in processing with pic: {e}"
    

import os
from datetime import datetime
import cv2

def SCORE_TIME_DETECTION(query, material):
    """
    Detect timestamp and scoreboard information in football broadcast footage
    
    Args:
        query: Query prompt
        material: List containing file paths (typically length 1)
    
    Returns:
        Text result from VLM model
    """
    if not material or len(material) == 0:
        return "Error: No material provided"
    
    file_path = material[0]
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = file_path
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return "Error: Failed to extract middle frame from video"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = os.path.join(PROJECT_PATH, "log/cache")
        os.makedirs(cache_dir, exist_ok=True)
        image_path = os.path.join(cache_dir, f"SCORE_TIME_DETECTION_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
    else:
        return "Error: Unsupported file format"
    
    timestamp_info = extract_timestamp(image_path)
    
    vlm_prompt = f"We first used easyocr to analyze this football footage and obtained: {timestamp_info}. {query}"
    
    vlm_result = VLM(vlm_prompt, [image_path])
    
    return vlm_result
