from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os
from project_path import PROJECT_PATH


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def _ensure_output_dir():
    shot_path = f"{PROJECT_PATH}/log/clip_tmp"
    try:
        os.makedirs(shot_path, exist_ok=True)
        test_path = os.path.join(shot_path, '.write_test')
        with open(test_path, 'w') as f:
            f.write('ok')
        os.remove(test_path)
        return shot_path
    except Exception:
        pass
    tmp_path = "/tmp/clip_tmp"
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def _get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / fps) if total_frames > 0 else 0.0
        return fps, total_frames, width, height, duration
    finally:
        cap.release()


def _open_writer(out_path, fps, width, height):
    for fourcc in ["mp4v", "avc1", "H264"]:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _open_writer_avi(out_path, fps, width, height):
    for fourcc in ["XVID", "MJPG"]:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _write_clip_cv2(video_path, start_time, end_time, out_path):
    info = _get_video_info(video_path)
    if not info:
        return False
    fps, total_frames, width, height, duration = info
    if width <= 0 or height <= 0:
        return False
    if end_time <= start_time:
        return False

    start_frame = max(0, int(start_time * fps))
    end_frame = int(end_time * fps)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames)
    if end_frame - start_frame < 1:
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    def _write_fallback_image():
        img_path = os.path.splitext(out_path)[0] + ".jpg"
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        if cv2.imwrite(img_path, frame):
            return img_path
        return False

    writer = _open_writer(out_path, fps, width, height)
    used_path = out_path
    if writer is None:
        alt_path = os.path.splitext(out_path)[0] + ".avi"
        writer = _open_writer_avi(alt_path, fps, width, height)
        used_path = alt_path if writer is not None else out_path
    if writer is None:
        result = _write_fallback_image()
        cap.release()
        return result if result else False

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame
        while current < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            current += 1
    finally:
        writer.release()
        cap.release()

    if used_path != out_path:
        # if we used avi, report success but caller should record actual path
        return used_path
    return True

def _fallback_start_end_clips(video_path, fallback_ratio=0.10, min_seconds=1.0, max_seconds=5.0):
    info = _get_video_info(video_path)
    if not info:
        return []
    fps, total_frames, width, height, duration = info
    if duration <= 0:
        return []

    clip_len = duration * fallback_ratio
    clip_len = max(min_seconds, clip_len)
    clip_len = min(max_seconds, clip_len)
    clip_len = min(clip_len, duration / 2.0)
    if clip_len <= 0:
        return []

    output_dir = os.path.join(_ensure_output_dir(), "fallback")
    os.makedirs(output_dir, exist_ok=True)
    start_path = f"{output_dir}/fallback_start.mp4"
    end_path = f"{output_dir}/fallback_end.mp4"

    out_paths = []
    result = _write_clip_cv2(video_path, 0.0, clip_len, start_path)
    if result:
        out_paths.append(result if isinstance(result, str) else start_path)
    result = _write_clip_cv2(video_path, max(0.0, duration - clip_len), duration, end_path)
    if result:
        out_paths.append(result if isinstance(result, str) else end_path)
    return out_paths


def SHOT_CHANGE(query=None, material=[], fallback_ratio=0.10, min_seconds=1.0, max_seconds=5.0):
    def _format_fallback(paths):
        return f"Shot change detection completed. The clips are saved in {paths}."

    def _fallback_or_original(video_path, error_msg=None):
        fallback_paths = []
        try:
            fallback_paths = _fallback_start_end_clips(
                video_path,
                fallback_ratio=fallback_ratio,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            )
        except Exception as exc:
            if not error_msg:
                error_msg = f"{type(exc).__name__}: {exc}"

        if fallback_paths:
            return _format_fallback(fallback_paths)
        if error_msg:
            return (
                f"Shot change detection failed ({error_msg}). "
                f"Fallback to original video clip: [{video_path}]."
            )
        return f"No scene changes detected in the video. Fallback to original video clip: [{video_path}]."

    if len(material) == 0:
        return "No video material provided."
    if len(material) > 1:
        return "Only one video material is supported at a time. But you provided more than one."
    video_path = material[0]
    if not is_video_file(video_path):
        return "The provided file is not a valid video file."

    if _env_flag("SHOT_CHANGE_FORCE_FALLBACK", default=False):
        return _fallback_or_original(
            video_path,
            "Forced fallback by SHOT_CHANGE_FORCE_FALLBACK=1",
        )

    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        scene_list = []
        error_msg = None
        try:
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(video_manager)
            scene_list = scene_manager.get_scene_list()
        except Exception as e:
            error_msg = str(e)
        finally:
            try:
                video_manager.release()
            except Exception:
                pass
    except Exception as exc:
        return _fallback_or_original(video_path, f"{type(exc).__name__}: {exc}")

    if len(scene_list) == 0:
        return _fallback_or_original(video_path, error_msg)

    output_path = []
    change_time = []
    output_dir = _ensure_output_dir()
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        change_time.append(end_time)
        new_video_path = f"{output_dir}/scene_{i+1}.mp4"
        result = _write_clip_cv2(video_path, start_time, end_time, new_video_path)
        if result:
            output_path.append(result if isinstance(result, str) else new_video_path)

    if not output_path:
        return _fallback_or_original(video_path, error_msg)

    return (
        f"Shot change detection completed. {len(scene_list)} scenes detected. "
        f"The clips are saved in {output_path}. Change occurred at {change_time[:-1]} seconds."
    )
