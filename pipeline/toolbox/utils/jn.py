import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .all_devices import get_qwen_device_map
from torch.backends import cudnn
import math
from project_path import PROJECT_PATH

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_PATH, ".env"))
except Exception:
    pass

_LEGIBILITY_CACHE = {}
_QWEN_CACHE = {}


def _device_key(device) -> str:
    try:
        return str(device)
    except Exception:
        return repr(device)

def _get_shared_vlm(model_name: str):
    if os.getenv("JERSEY_USE_SHARED_VLM", "1") == "0":
        return None, None
    try:
        from . import vlm_distribution
        shared_name = getattr(vlm_distribution, "VLM_MODEL_NAME", None)
        if shared_name and str(shared_name) == str(model_name):
            return getattr(vlm_distribution, "vlm_model", None), getattr(vlm_distribution, "vlm_processor", None)
    except Exception:
        return None, None
    return None, None


def get_legibility_model(model_path, device):
    key = (str(model_path), _device_key(device))
    if key not in _LEGIBILITY_CACHE:
        _LEGIBILITY_CACHE[key] = Legibility(model_path, device)
    return _LEGIBILITY_CACHE[key]


def get_qwen_ocr_model(qwen_path, device):
    env_model = os.getenv("VLM_MODEL_NAME")
    if env_model:
        qwen_path = env_model
    key = (str(qwen_path), _device_key(device))
    if key not in _QWEN_CACHE:
        _QWEN_CACHE[key] = QWEN2_5VL_OCR_BATCH(qwen_model_path=qwen_path, device=device)
    return _QWEN_CACHE[key]


def clear_model_cache():
    """Clear cached legibility/Qwen models to free memory."""
    _LEGIBILITY_CACHE.clear()
    _QWEN_CACHE.clear()

class LegibilityClassifier34(nn.Module):
    def __init__(self, train=False,  finetune=False):
        super().__init__()
        self.model_ft = models.resnet34(pretrained=True)
        if finetune:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)
        self.model_ft.fc.requires_grad = True
        self.model_ft.layer4.requires_grad = True

    def forward(self, x):
        x = self.model_ft(x)
        x = F.sigmoid(x)
        return x


class Legibility():
    def __init__(self, legibility_model_path, device):
        self.device = device
        cudnn.benchmark = True

        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize model
        self.model = LegibilityClassifier34()
        state_dict = torch.load(legibility_model_path, map_location=device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def process(self, image_list, threshold=0.5):
        img_crops_PIL = image_list
        img_crops_PIL_transformed = [self.transforms(img_crop_PIL) for img_crop_PIL in img_crops_PIL]
        img_crops_PIL_transformed = torch.stack(img_crops_PIL_transformed).to(self.device)

        # Get legibility scores
        outputs = self.model(img_crops_PIL_transformed)
        if threshold > 0:
            outputs = (outputs>threshold).float()
        else:
            outputs = outputs.float()
        legibility_scores = outputs[:,0].float().cpu().detach().numpy()

        return list(legibility_scores)


class QWEN2_5VL_OCR_BATCH():
    def __init__(self, qwen_model_path, device):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_path = qwen_model_path
        self.save_jersey_number_full_detection = True
        self.use_legibility_filter = True
        self._shared_vlm = False

        shared_model, shared_processor = _get_shared_vlm(self.model_path)
        if shared_model is not None:
            self.model = shared_model
            self.processor = shared_processor or AutoProcessor.from_pretrained(self.model_path)
            self.device = self.model.device
            self._shared_vlm = True
            self._orig_padding_side = getattr(self.processor.tokenizer, "padding_side", None) if hasattr(self.processor, "tokenizer") else None
        else:
            device_map_kwargs = get_qwen_device_map(reserve_first_gpu=True, reserve_first_n=1) if torch.cuda.is_available() else {}
            if "Qwen3-VL" in str(self.model_path):
                from transformers import Qwen3VLForConditionalGeneration
                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                }
                model_kwargs.update(device_map_kwargs)
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            else:
                model_kwargs = {
                    "dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                }
                model_kwargs.update(device_map_kwargs)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
            self.device = device

        self.text_prompt = "Analyze this image and determine if the player is facing away from the camera. If the player is facing away, output the jersey number on their back. If the player is not facing away from the camera, output 'No'."

    def no_jersey_number(self):
        return None, 0

    def extract_numbers(self, text):
        if text.strip() == "?":
            return None
        # Strip thinking block if present (<think>...</think>)
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Find first standalone 1-3 digit number (jersey numbers are typically ≤99)
        m = re.search(r'\b(\d{1,3})\b', clean)
        if m:
            return m.group(1)
        return None

    @torch.no_grad()
    def process(self, batch, threshold=0.5, batch_size=64):
        real_bs = len(batch['imgs'])
        jersey_number_detection = [None] * real_bs
        jersey_number_confidence = [0.0] * real_bs
        jersey_number_full_detection = [''] * real_bs

        # Create a list of valid indices based on legibility filter
        idxs = []
        if self.use_legibility_filter:
            for i, score in enumerate(batch['legibility_score']):
                if score >= threshold:
                    idxs.append(i)
        else:
            idxs = list(range(len(batch['imgs'])))

        sampled_idxs = []
        stride = None

        if len(idxs) > 0:
            # Downsample legible frames so Qwen sees ~25-26 representative images instead of every frame
            stride = max(1, math.ceil(len(idxs) / 25))
            sampled_idxs = idxs[::stride][:26]

            print(f"Processing {len(sampled_idxs)} images (sampled from {len(idxs)} legible frames, stride={stride})")
            batch_imgs = [batch['imgs'][idx] for idx in sampled_idxs]

            # Single multi-image prompt so the model can aggregate across frames.
            multi_image_prompt = (
                "You are given multiple images of the same player from different frames. "
                "Look across ALL images together. If you see a jersey number on the player's back, "
                "output only that number. If no clear jersey number is visible in any image, output 'No'."
            )

            messages = [[
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": img} for img in batch_imgs] +
                        [{"type": "text", "text": multi_image_prompt}]
                    )
                }
            ]]

            text = self.processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            image_inputs, video_inputs = process_vision_info(messages)

            restore_pad = None
            if self._shared_vlm and hasattr(self.processor, "tokenizer"):
                restore_pad = self.processor.tokenizer.padding_side
                self.processor.tokenizer.padding_side = "left"
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
            finally:
                if restore_pad is not None:
                    self.processor.tokenizer.padding_side = restore_pad

            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            jersey_number = self.extract_numbers(output_text)

            # Fill the prediction back to all legible frames (not just sampled) so the downstream
            # majority vote / consecutive filter can work with dense indices.
            target_idxs = idxs if idxs else list(range(len(batch['imgs'])))
            for idx in target_idxs:
                jersey_number_detection[idx] = jersey_number
                jersey_number_confidence[idx] = 1.0 if jersey_number is not None else 0.0
                if self.save_jersey_number_full_detection:
                    jersey_number_full_detection[idx] = output_text
            print(f"Multi-image -> {jersey_number if jersey_number is not None else 'None'}")

        # Expose indices/stride for debugging downstream
        batch['legible_idxs'] = idxs
        batch['sampled_idxs'] = sampled_idxs
        batch['sample_stride'] = stride
        batch['multi_image_number'] = jersey_number if len(idxs) > 0 else None
        batch['multi_image_text'] = output_text if len(idxs) > 0 else ""
        batch['jersey_number_detection'] = jersey_number_detection
        batch['jersey_number_confidence'] = jersey_number_confidence
        if self.save_jersey_number_full_detection:
            batch['jersey_number_full_detection'] = jersey_number_full_detection

        return batch


class MajorityVoteTrackletFilter():
    def __init__(self):
        pass

    def select_highest_voted_att(self, values, confidences):
        value_counts = {}
        for v, c in zip(values, confidences):
            if v is not None:
                if v not in value_counts:
                    value_counts[v] = 0
                value_counts[v] += c

        if len(value_counts) == 0:
            return None

        max_value = max(value_counts, key=value_counts.get)
        return max_value

    @torch.no_grad()
    def process(self, tracklet):
        attribute_detection = tracklet['jersey_number_detection']
        attribute_confidence = tracklet['jersey_number_confidence']

        # Convert to list for easier manipulation
        detection_list = list(attribute_detection)
        confidence_list = list(attribute_confidence)

        # Remove None values and vote directly
        final_detection = []
        final_confidence = []
        for d, c in zip(detection_list, confidence_list):
            if d is not None:
                final_detection.append(d)
                final_confidence.append(c)

        # Get majority vote from filtered values
        if final_detection:
            attribute_value = self.select_highest_voted_att(final_detection, final_confidence)
            print(f"\n📊 Majority Vote Results:")
            from collections import Counter
            counts = Counter(final_detection)
            for num, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {num}: {count} votes")
            print(f"   Winner: {attribute_value}")
        else:
            attribute_value = None
            print(f"\n⚠️ No valid detections to vote on")

        tracklet['jn_final'] = [attribute_value] * len(detection_list)

        return attribute_value, tracklet


def run(device, img_list, model_path, qwen_path, threshold=0.5):
    if len(img_list) == 1:
        # Single image: relax legibility filtering so OCR still runs.
        threshold = 0.0
    lc = get_legibility_model(model_path, device)
    lc_score = lc.process(img_list, threshold)
    # print(lc_score)

    qwen = get_qwen_ocr_model(qwen_path, device)
    results = qwen.process({"imgs": img_list, "legibility_score": lc_score}, threshold)
    # If multi-image result exists, trust it as final answer
    multi_ans = results.get("multi_image_number")
    if multi_ans is not None:
        results["jn_final"] = [multi_ans] * len(img_list)
        return multi_ans, results

    # Fallback to majority vote
    filter = MajorityVoteTrackletFilter()
    ans, results = filter.process(results)
    return ans, results
