import os
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from project_path import PROJECT_PATH

try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc

# NumPy 1.24+ compatibility (some insightface versions use np.int)
if not hasattr(np, "int"):
    np.int = int

_APP = None
_APP_CFG = None


def _configure_opencv_logging():
    """Reduce noisy OpenCV/libpng warnings in long batch jobs."""
    try:
        # OpenCV >= 4.8 style
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        return
    except Exception:
        pass
    try:
        # Some builds expose setLogLevel directly
        cv2.setLogLevel(2)  # 2: ERROR
    except Exception:
        pass


_configure_opencv_logging()


def _largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(faces, key=area)


def _resolve_image_path(path):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_PATH, path)
    if os.path.exists(candidate):
        return candidate
    return path


def _l2norm(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def _aggregate_with_closest_k(
    encodings,
    keep_closest_k: int = 3,
    min_samples: int = 4,
    enabled: bool = True,
):
    """
    Aggregate per-identity embeddings by keeping the closest-k vectors to the
    global mean embedding, then averaging them.

    Returns:
        aggregated_embedding, used_count, total_count, strategy_name
    """
    encs = np.asarray(encodings, dtype=np.float32)
    total_count = int(encs.shape[0])
    if total_count == 0:
        raise ValueError("encodings must not be empty")

    # First-pass center from all samples.
    center = _l2norm(np.mean(encs, axis=0))

    # Protect small-sample identities: keep all when there are fewer than 4 samples.
    if (not enabled) or total_count < min_samples:
        agg = _l2norm(np.mean(encs, axis=0))
        strategy = "all_mean" if enabled else "all_mean_disabled"
        return agg, total_count, total_count, strategy

    k = max(1, min(int(keep_closest_k), total_count))
    # Cosine distance on normalized embeddings: smaller means closer to center.
    dists = 1.0 - np.clip(np.dot(encs, center), -1.0, 1.0)
    keep_idx = np.argsort(dists)[:k]
    kept = encs[keep_idx]

    # Final representative embedding after trimming far samples.
    agg = _l2norm(np.mean(kept, axis=0))
    return agg, int(k), total_count, "closest_k_mean"


def _get_app(
    det_size: Tuple[int, int] = (640, 640),
    ctx_id: int = 0,
    providers: Optional[Sequence[str]] = None,
):
    global _APP, _APP_CFG

    if FaceAnalysis is None:
        raise ImportError(
            f"InsightFace is not available: {_INSIGHTFACE_IMPORT_ERROR}"
        )

    det_size = tuple(det_size)
    cfg = (det_size, ctx_id, tuple(providers) if providers else None)
    if _APP is not None and _APP_CFG == cfg:
        return _APP, ctx_id

    # Explicitly pass device_id in provider options so onnxruntime is pinned to
    # the correct GPU regardless of PyTorch's current CUDA device context.
    # Without this, if the VLM has shifted the current CUDA device (e.g. to GPU=1
    # for its first shard), onnxruntime will inherit that device and conflict with
    # VLM memory, causing CUDA illegal memory access errors.
    providers_try = list(providers) if providers else [
        ("CUDAExecutionProvider", {"device_id": ctx_id}),
        "CPUExecutionProvider",
    ]

    try:
        app = FaceAnalysis(providers=providers_try)
        app.prepare(ctx_id=ctx_id, det_size=det_size)
    except Exception:
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        ctx_id = -1
        app.prepare(ctx_id=ctx_id, det_size=det_size)

    _APP = app
    _APP_CFG = (det_size, ctx_id, tuple(providers_try) if providers else None)
    return app, ctx_id


def _embed_image(path: Path, app) -> Optional[np.ndarray]:
    # Prefer PIL path for PNG to avoid noisy libpng profile warnings in OpenCV.
    path = Path(path)
    if path.suffix.lower() == ".png":
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = app.get(img)
    best = _largest_face(faces)
    if best is None:
        return None
    return _l2norm(best.embedding)


def _save_face_library(face_library, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(face_library, f)


def _default_face_lib_path():
    arc = f"{PROJECT_PATH}/pipeline/toolbox/utils/face_library_arc.pkl"
    if os.path.exists(arc):
        return arc
    return f"{PROJECT_PATH}/pipeline/toolbox/utils/face_library.pkl"


def build_face_library(
    base_path,
    output_path=None,
    model="arcface",
    resume=True,
    save_every=1,
    det_size=(640, 640),
    max_per_id: Optional[int] = None,
    ctx_id: int = 0,
    providers: Optional[Sequence[str]] = None,
    keep_closest_k: int = 3,
    closest_k_min_samples: int = 4,
    closest_k_enabled: bool = True,
):
    """Build a face embedding library with per-player ArcFace embeddings.

    Args:
        base_path: path to directory containing per-person subfolders of images.
        output_path: pickle path. Defaults to face_library_arc.pkl.
        model: kept for compatibility (unused; ArcFace/RetinaFace via InsightFace).
        resume: load existing library and skip already processed identities.
        save_every: save progress every N identities.
        det_size: detection size for RetinaFace.
        max_per_id: limit number of images per identity.
        keep_closest_k: number of embeddings to keep closest to the mean center.
        closest_k_min_samples: apply closest-k only when sample count >= this value.
        closest_k_enabled: enable/disable closest-k aggregation.
    """
    valid_image_extensions = {".jpg", ".jpeg", ".png"}
    face_library = {}
    if output_path is None:
        output_path = f"{PROJECT_PATH}/pipeline/toolbox/utils/face_library_arc.pkl"

    app, _ = _get_app(det_size=det_size, ctx_id=ctx_id, providers=providers)

    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                face_library = loaded
                print(
                    f"Resuming from {output_path} with {len(face_library)} existing entries."
                )
        except Exception as e:
            print(f"Failed to load existing face library ({output_path}): {e}")

    person_folders = sorted(
        [
            folder
            for folder in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, folder))
        ]
    )

    processed_count = 0
    for person_folder in person_folders:
        if resume and person_folder in face_library:
            continue
        person_folder_path = os.path.join(base_path, person_folder)
        img_files = sorted(
            [
                file
                for file in os.listdir(person_folder_path)
                if os.path.splitext(file)[1].lower() in valid_image_extensions
            ]
        )

        if max_per_id:
            img_files = img_files[:max_per_id]

        encodings = []
        for img_file in img_files:
            img_path = Path(person_folder_path) / img_file
            try:
                emb = _embed_image(img_path, app)
            except Exception as e:
                print(f"Skipped {img_path}: {e}")
                continue
            if emb is not None:
                encodings.append(emb)

        if encodings:
            # Robust aggregation: keep closest-k embeddings to reduce outlier influence.
            agg_emb, used_count, total_count, strategy = _aggregate_with_closest_k(
                encodings,
                keep_closest_k=keep_closest_k,
                min_samples=closest_k_min_samples,
                enabled=closest_k_enabled,
            )
            face_library[person_folder] = agg_emb
            if strategy == "closest_k_mean":
                print(
                    f"Added {person_folder} to the face library. "
                    f"({used_count}/{total_count} faces kept by closest-k)"
                )
            elif strategy == "all_mean_disabled":
                print(
                    f"Added {person_folder} to the face library. "
                    f"({total_count} faces, closest-k disabled)"
                )
            else:
                print(
                    f"Added {person_folder} to the face library. "
                    f"({total_count} faces, closest-k skipped for small sample)"
                )

        processed_count += 1
        if resume and save_every and processed_count % save_every == 0:
            _save_face_library(face_library, output_path)

    _save_face_library(face_library, output_path)
    print(f"Face library saved to {output_path}.")

    return face_library


def build_arc_face_library(
    base_path: Path,
    output_path: Path,
    resume: bool = True,
    save_every: int = 200,
    max_per_id: Optional[int] = None,
    det_size=(640, 640),
    ctx_id: int = 0,
    providers: Optional[Sequence[str]] = None,
    keep_closest_k: int = 3,
    closest_k_min_samples: int = 4,
    closest_k_enabled: bool = True,
):
    """
    Notebook-friendly wrapper around build_face_library.
    Accepts Path inputs and keeps argument names used in experiment notebooks.
    """
    return build_face_library(
        base_path=str(base_path),
        output_path=str(output_path),
        resume=resume,
        save_every=save_every,
        max_per_id=max_per_id,
        det_size=det_size,
        ctx_id=ctx_id,
        providers=providers,
        keep_closest_k=keep_closest_k,
        closest_k_min_samples=closest_k_min_samples,
        closest_k_enabled=closest_k_enabled,
    )


def FACE_RECOGNITION(
    query=None,
    material=None,
    agg="min",
    model="arcface",
    det_size=(640, 640),
    library_path: Optional[str] = None,
    ctx_id: int = 0,
    providers: Optional[Sequence[str]] = None,
):
    """Identify person by ArcFace embeddings using InsightFace.

    Args:
        material: list of image paths.
        agg: aggregation across multiple query images (min or mean distance).
        library_path: custom pickle path. Defaults to face_library_arc.pkl if exists.
    """
    filename = library_path or _default_face_lib_path()
    if not os.path.exists(filename):
        return "None"

    with open(filename, "rb") as f:
        face_library = pickle.load(f)

    if not material:
        return "None"

    app, _ = _get_app(det_size=det_size, ctx_id=ctx_id, providers=providers)

    query_encodings = []
    for item in material:
        img_path = _resolve_image_path(item)
        try:
            emb = _embed_image(Path(img_path), app)
        except Exception:
            continue
        if emb is not None:
            query_encodings.append(emb)

    if not query_encodings:
        return "None"

    names = list(face_library.keys())
    encs = np.vstack([face_library[name] for name in names])

    def cosine_distance(a, b):
        return 1.0 - float(np.dot(a, b))

    dist_matrix = np.stack(
        [np.array([cosine_distance(le, qe) for le in encs]) for qe in query_encodings]
    )

    if agg == "min":
        scores = dist_matrix.min(axis=0)
    elif agg == "mean":
        scores = dist_matrix.mean(axis=0)
    else:
        raise ValueError("agg must be 'min' or 'mean'")

    best_idx = int(scores.argmin())
    best_name = names[best_idx]
    best_score = scores[best_idx]
    return f"The person in the photo is most likely: {best_name}, distance: {best_score}"
