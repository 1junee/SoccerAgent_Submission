import os
from pathlib import Path
import json
from .utils.GroundingDINO.demo.inference_on_a_image import run_inference_on_image
import cv2
from project_path import PROJECT_PATH


def _resolve_input_path(input_path: str):
    if not input_path:
        return None, "No material provided."
    path = Path(input_path)
    if path.is_dir():
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
        candidates = sorted([p for p in path.iterdir() if p.suffix.lower() in image_exts])
        if not candidates:
            return None, f"No image files found in directory: {input_path}"
        return str(candidates[0]), None
    if not path.exists():
        return None, f"Material not found: {input_path}"
    return str(path), None


def SEGMENT(query=None, material=[]):
    config_file = f"{PROJECT_PATH}/pipeline/toolbox/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    model_weights = f"{PROJECT_PATH}/pipeline/toolbox/utils/GroundingDINO/groundingdino/config/groundingdino_swinb_cogcoor.pth"
    gdino_device = os.getenv("GDINO_DEVICE", "cuda:0")
    image_path = material[0] if material else None
    text = query
    output_path = f"{PROJECT_PATH}/log/segment" # replace with your helper file path to save the output image
    os.makedirs(output_path, exist_ok=True)
    descriptions = "No valid material provided."

    image_path, error = _resolve_input_path(image_path)
    if error:
        return error

    file_extension = os.path.splitext(image_path)[1].lower()
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']

    if file_extension in image_extensions:
        pred_dict = run_inference_on_image(config_file, model_weights, image_path, text, output_path, device=gdino_device)
        output_img = os.path.join(output_path, "pred.jpg")
        if os.path.exists(output_img):
            descriptions = f"The object you want to segment has been prompted with bounding box on the image, which is saved at [{output_img}]."
        else:
            descriptions = "The object you want to segment isn't detected in the image."

    elif file_extension in video_extensions:
        video_path = image_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        images_path = []
        frame_count = 0
        stride = 10
        frame_path = os.path.join(output_path, 'frames')
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % stride != 0:
                continue
            frame_filename = os.path.join(frame_path, f"frame_{frame_count:04d}.jpg")

            cv2.imwrite(frame_filename, frame)
            images_path.append(frame_filename)
        cap.release()
        max_detections = -1
        best_image_path = None
        best_pred_dict = None

        for image_path in images_path:
            pred_dict = run_inference_on_image(config_file, model_weights, image_path, text, output_path, device=gdino_device)
            current_detections = len(pred_dict['boxes'])

            if current_detections > max_detections:
                max_detections = current_detections
                best_image_path = image_path
                best_pred_dict = pred_dict
        if best_pred_dict is not None:
            pred_dict = run_inference_on_image(config_file, model_weights, best_image_path, text, output_path, device=gdino_device)
            output_img = os.path.join(output_path, "pred.jpg")
            if os.path.exists(output_img):
                descriptions = f"The object you want to segment has been prompted with bounding box on the image, which is saved at [{output_img}]."
            else:
                descriptions = "The object you want to segment isn't detected in the video."
        else:
            descriptions = "The object you want to segment isn't detected in the video."
    else:
        descriptions = f"Unsupported file type: {file_extension}"
    return descriptions
