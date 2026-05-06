"""A.I.R. Frame Selection
========================
Adaptive, Iterative, Reasoning-based Frame Selection (arxiv: 2510.04428)

Stage 1: CLIP-based event detection + adaptive initial sampling
Stage 2: Iterative VLM reasoning + localized density sampling
Stage 3: return final frame paths (QA handled by caller)

All parameters configurable via AIR_* environment variables.
Falls back to uniform sampling on any failure.
"""

import gc
import logging
import math
import os
import re
import uuid

import cv2
import ffmpeg
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ── module-level CLIP cache ───────────────────────────────────────────────────
_CLIP_MODEL     = None
_CLIP_PROCESSOR = None
_CLIP_MODEL_ID  = None
_CLIP_DEVICE    = None


def _get_clip(model_name: str, device: str):
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_MODEL_ID, _CLIP_DEVICE
    if (
        _CLIP_MODEL is None
        or _CLIP_MODEL_ID != model_name
        or _CLIP_DEVICE != device
    ):
        if _CLIP_MODEL is not None:
            del _CLIP_MODEL, _CLIP_PROCESSOR
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device)
        _CLIP_MODEL.eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _CLIP_MODEL_ID = model_name
        _CLIP_DEVICE = device
    return _CLIP_MODEL, _CLIP_PROCESSOR


def _select_device() -> str:
    env = os.getenv("FRAME_SELECTION_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


class AIRFrameSelector:
    """
    A.I.R.: Adaptive, Iterative, Reasoning-based Frame Selection.

    Paper: https://arxiv.org/abs/2510.04428

    Usage::

        selector = AIRFrameSelector()
        frame_paths = selector.select_frames(
            video_path="video.mp4",
            query="What foul did the player commit?",
            vlm_fn=lambda text, imgs: model.chat_img(text, imgs, max_tokens=256),
            tmp_dir="/tmp/air_frames/item_001",
        )
        # frame_paths: list of JPEG file paths — caller is responsible for cleanup.
    """

    def __init__(
        self,
        clip_model_name: str | None = None,
        analysis_fps: float | None = None,
        budget_min: int | None = None,
        budget_max: int | None = None,
        budget_per_sec: float | None = None,
        gamma: float | None = None,
        d_min: int | None = None,
        l_min: int | None = None,
        i_max: int | None = None,
        c_intervals: int | None = None,
        theta: float | None = None,
        alpha: int | None = None,
        beta: float | None = None,
        d_lds: int | None = None,
        c_len: float | None = None,
    ):
        self.clip_model_name = clip_model_name or os.getenv(
            "AIR_CLIP_MODEL", "openai/clip-vit-large-patch14"
        )
        self.analysis_fps   = analysis_fps   if analysis_fps   is not None else _env_float("AIR_ANALYSIS_FPS",   2.0)
        self.budget_min     = budget_min     if budget_min     is not None else _env_int  ("AIR_BUDGET_MIN",     8)
        self.budget_max     = budget_max     if budget_max     is not None else _env_int  ("AIR_BUDGET_MAX",     32)
        self.budget_per_sec = budget_per_sec if budget_per_sec is not None else _env_float("AIR_BUDGET_PER_SEC", 1.0)
        self.gamma          = gamma          if gamma          is not None else _env_float("AIR_GAMMA",          0.5)
        self.d_min          = d_min          if d_min          is not None else _env_int  ("AIR_D_MIN",          8)
        self.l_min          = l_min          if l_min          is not None else _env_int  ("AIR_L_MIN",          4)
        self.i_max          = i_max          if i_max          is not None else _env_int  ("AIR_IMAX",           3)
        self.c_intervals    = c_intervals    if c_intervals    is not None else _env_int  ("AIR_C",              3)
        self.theta          = theta          if theta          is not None else _env_float("AIR_THETA",          3.0)
        self.alpha          = alpha          if alpha          is not None else _env_int  ("AIR_ALPHA",          2)
        self.beta           = beta           if beta           is not None else _env_float("AIR_BETA",           2.0)
        self.d_lds          = d_lds          if d_lds          is not None else _env_int  ("AIR_D_LDS",          4)
        self.c_len          = c_len          if c_len          is not None else _env_float("AIR_CLEN",           0.5)

    # ── frame extraction ──────────────────────────────────────────────────────

    def _extract_frames(self, video_path: str) -> list[tuple[int, float, Image.Image]]:
        """Extract frames via ffmpeg rawvideo pipe at analysis_fps."""
        probe = ffmpeg.probe(video_path)
        vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
        w, h = int(vs["width"]), int(vs["height"])
        sar = vs.get("sample_aspect_ratio", "1:1") or "1:1"
        if sar not in ("1:1", "0:1", "N/A"):
            parts = sar.split(":")
            if len(parts) == 2:
                n, d = int(parts[0]), int(parts[1])
                if d != 0:
                    w = round(w * n / d)

        process = (
            ffmpeg.input(video_path)
            .filter("scale", w, h)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", r=self.analysis_fps)
            .run_async(pipe_stdout=True, quiet=True)
        )

        frames: list[tuple[int, float, Image.Image]] = []
        idx = 0
        while True:
            raw = process.stdout.read(w * h * 3)
            if not raw:
                break
            img = Image.frombytes("RGB", (w, h), raw)
            frames.append((idx, idx / self.analysis_fps, img.copy()))
            idx += 1
        process.wait()
        return frames

    # ── CLIP similarity ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_clip_sim(
        self,
        query: str,
        frames: list[tuple[int, float, Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Compute query-frame cosine similarity. Returns (N,) float32."""
        device = _select_device()
        model, processor = _get_clip(self.clip_model_name, device)

        text_inputs = processor(
            text=[query], return_tensors="pt", truncation=True, max_length=77
        )
        text_feat = model.get_text_features(
            **{k: v.to(device) for k, v in text_inputs.items()}
        )
        if not isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.pooler_output
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_np = text_feat.detach().cpu().float().numpy()  # (1, D)

        images = [f[2] for f in frames]
        sims_parts = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            img_inputs = processor(images=batch, return_tensors="pt")
            img_feat = model.get_image_features(
                pixel_values=img_inputs["pixel_values"].to(device)
            )
            if not isinstance(img_feat, torch.Tensor):
                img_feat = img_feat.pooler_output
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_np = img_feat.detach().cpu().float().numpy()       # (B, D)
            sims_parts.append((img_np * text_np).sum(axis=1))  # (B,)
            del img_feat, img_np, img_inputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(sims_parts, axis=0)

    # ── GMM threshold ─────────────────────────────────────────────────────────

    def _gmm_threshold(self, sims: np.ndarray) -> float:
        """Fit 2-component GMM and compute relevance threshold T."""
        from sklearn.mixture import GaussianMixture

        X = sims.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(X)
        means  = gmm.means_.flatten()
        covars = np.sqrt(gmm.covariances_.flatten())
        T = float(means.max() - self.gamma * covars.max())
        return T

    # ── event detection ───────────────────────────────────────────────────────

    def _detect_events(self, sims: np.ndarray, T: float) -> list[tuple[int, int]]:
        """Find contiguous regions with sims > T; merge/prune."""
        N = len(sims)
        active = sims > T

        events: list[list[int]] = []
        in_event = False
        start = 0
        for i in range(N):
            if active[i] and not in_event:
                start = i
                in_event = True
            elif not active[i] and in_event:
                events.append([start, i - 1])
                in_event = False
        if in_event:
            events.append([start, N - 1])

        # merge gaps < d_min
        merged: list[list[int]] = []
        for ev in events:
            if merged and ev[0] - merged[-1][1] < self.d_min:
                merged[-1][1] = ev[1]
            else:
                merged.append(ev)

        # prune short events
        pruned = [(s, e) for s, e in merged if e - s + 1 >= self.l_min]
        return pruned

    # ── initial sampling ──────────────────────────────────────────────────────

    def _initial_sampling(
        self,
        sims: np.ndarray,
        events: list[tuple[int, int]],
        budget: int,
    ) -> list[int]:
        """Allocate frames proportionally to event duration using peak-sim selection."""
        total_len = sum(e - s + 1 for s, e in events)
        if total_len == 0:
            return []

        selected: list[int] = []
        for s, e in events:
            length = e - s + 1
            n_frames = max(1, round(budget * length / total_len))
            spacing = max(1, length // (n_frames + 1))

            remaining = list(range(s, e + 1))
            for _ in range(n_frames):
                if not remaining:
                    break
                best_idx = max(remaining, key=lambda i: sims[i])
                selected.append(best_idx)
                remaining = [i for i in remaining if abs(i - best_idx) > spacing]

        return sorted(set(selected))

    # ── potential ranking ─────────────────────────────────────────────────────

    def _compute_potential(
        self, sims: np.ndarray, sel_indices: list[int]
    ) -> list[tuple[int, int, float]]:
        """
        For each interval between consecutive selected frames compute:
            Potential = Mean(S) × (1 + Σ|ΔS|/len) × (1 + c_len·ln(len))
        Returns list of (i, j, potential) sorted descending.
        """
        if len(sel_indices) < 2:
            return []

        intervals = []
        for a, b in zip(sel_indices[:-1], sel_indices[1:]):
            if b <= a:
                continue
            segment = sims[a : b + 1]
            length  = len(segment)
            mean_s  = float(segment.mean())
            complexity = float(np.abs(np.diff(segment)).sum()) / length if length > 1 else 0.0
            len_factor = math.log(max(1, length))
            potential = mean_s * (1.0 + complexity) * (1.0 + self.c_len * len_factor)
            intervals.append((a, b, potential))

        intervals.sort(key=lambda x: -x[2])
        return intervals

    # ── frame saving ──────────────────────────────────────────────────────────

    def _save_frames(
        self,
        frames: list[tuple[int, float, Image.Image]],
        indices: list[int],
        tmp_dir: str,
    ) -> list[str]:
        """Save selected frames as JPEG to tmp_dir. Returns paths in index order."""
        os.makedirs(tmp_dir, exist_ok=True)
        paths = []
        for idx in sorted(set(indices)):
            if idx < 0 or idx >= len(frames):
                continue
            img = frames[idx][2]
            fpath = os.path.join(tmp_dir, f"frame_{idx:06d}.jpg")
            img.save(fpath, format="JPEG", quality=90)
            paths.append(fpath)
        return paths

    # ── VLM scoring ───────────────────────────────────────────────────────────

    def _vlm_score(
        self,
        query: str,
        frame_paths: list[str],
        vlm_fn,
    ) -> list[float]:
        """
        Ask vlm_fn to rate frame relevance 1-5.
        Returns list[float] of length len(frame_paths); defaults to 3.0 on failure.
        """
        n = len(frame_paths)
        if n == 0:
            return []

        prompt = (
            "You are evaluating video frame relevance for a football question.\n"
            f"Question: {query}\n\n"
            "Rate each frame's relevance to answering this question on a scale of 1 to 5:\n"
            "  1 = not relevant at all\n"
            "  3 = somewhat relevant\n"
            "  5 = directly relevant and helpful\n\n"
            "For each frame, give a brief reasoning and an integer score.\n"
            "Format exactly as:\n"
            "Frame 1: <reasoning> Score: X\n"
            "Frame 2: <reasoning> Score: X\n"
            "...\n"
            "Only output the frame evaluations, nothing else."
        )

        try:
            response = vlm_fn(prompt, frame_paths)
        except Exception as exc:
            logger.warning("[A.I.R.] VLM scoring call failed: %s", exc)
            return [3.0] * n

        scores = [3.0] * n
        pattern = re.compile(r"Frame\s*(\d+)[^S\n]*Score:\s*([1-5])", re.IGNORECASE)
        for m in pattern.finditer(response or ""):
            frame_num = int(m.group(1))
            score     = float(m.group(2))
            if 1 <= frame_num <= n:
                scores[frame_num - 1] = score
        return scores

    # ── localized density sampling ────────────────────────────────────────────

    def _lds(self, anchor: int, N: int, existing: set[int]) -> set[int]:
        """
        LDS(f*) = { f* ± α·β^(m-1) | m=1..D_lds }, clamped to [0, N-1].
        """
        new: set[int] = set()
        for m in range(1, self.d_lds + 1):
            stride = round(self.alpha * (self.beta ** (m - 1)))
            for sign in (-1, 1):
                candidate = anchor + sign * stride
                if 0 <= candidate < N and candidate not in existing:
                    new.add(candidate)
        return new

    # ── uniform fallback (cv2) ────────────────────────────────────────────────

    def _uniform_fallback_cv2(
        self, video_path: str, budget: int, tmp_dir: str
    ) -> list[str]:
        os.makedirs(tmp_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // budget)
        paths = []
        for i in range(0, total, step):
            if len(paths) >= budget:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            fpath = os.path.join(tmp_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(fpath, frame)
            paths.append(fpath)
        cap.release()
        return paths

    def _uniform_sample(
        self,
        frames: list[tuple[int, float, Image.Image]],
        budget: int,
        tmp_dir: str,
    ) -> list[str]:
        N = len(frames)
        if N == 0:
            return []
        step = max(1, N // budget)
        indices = list(range(0, N, step))[:budget]
        return self._save_frames(frames, indices, tmp_dir)

    # ── main entry point ──────────────────────────────────────────────────────

    def select_frames(
        self,
        video_path: str,
        query: str,
        vlm_fn,
        tmp_dir: str | None = None,
    ) -> list[str]:
        """
        A.I.R. frame selection pipeline.

        Args:
            video_path : path to video file
            query      : question text (CLIP text query + VLM scoring context)
            vlm_fn     : callable(prompt: str, image_paths: list[str]) -> str
            tmp_dir    : directory for temporary JPEG frames (created if needed)

        Returns:
            list of absolute JPEG file paths sorted by frame index.
            Caller is responsible for cleanup.
        """
        if tmp_dir is None:
            tmp_dir = os.path.join("tmp_file", "air_frames", uuid.uuid4().hex[:12])

        # ── adaptive budget ────────────────────────────────────────────────────
        try:
            probe = ffmpeg.probe(video_path)
            vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
            duration = float(vs.get("duration") or 0)
            if duration <= 0:
                duration = float(probe.get("format", {}).get("duration", 30))
        except Exception:
            duration = 30.0

        budget = int(
            min(
                self.budget_max,
                max(self.budget_min, round(self.budget_per_sec * duration)),
            )
        )
        print(
            f"[A.I.R.] {Path(video_path).name}  duration={duration:.1f}s  "
            f"budget={budget}  fps={self.analysis_fps}"
        )

        # ── Stage 1: extract + CLIP ────────────────────────────────────────────
        try:
            frames = self._extract_frames(video_path)
        except Exception as exc:
            logger.warning("[A.I.R.] ffmpeg extraction failed (%s) → cv2 fallback", exc)
            return self._uniform_fallback_cv2(video_path, budget, tmp_dir)

        N = len(frames)
        if N < 4:
            print("[A.I.R.] too few frames → uniform fallback")
            return self._save_frames(frames, list(range(N)), tmp_dir)

        try:
            sims = self._compute_clip_sim(query, frames)
        except Exception as exc:
            logger.warning("[A.I.R.] CLIP scoring failed (%s) → uniform fallback", exc)
            return self._uniform_sample(frames, budget, tmp_dir)

        # ── GMM threshold + events ─────────────────────────────────────────────
        try:
            T = self._gmm_threshold(sims)
        except Exception as exc:
            logger.warning("[A.I.R.] GMM failed (%s) → mean threshold", exc)
            T = float(sims.mean())

        events = self._detect_events(sims, T)
        print(f"[A.I.R.] GMM T={T:.4f}  events={len(events)}")

        if not events:
            print("[A.I.R.] no events → top-CLIP fallback")
            top_indices = np.argsort(-sims)[:budget].tolist()
            top_indices.sort()
            return self._save_frames(frames, top_indices, tmp_dir)

        sel = sorted(set(self._initial_sampling(sims, events, budget)))
        print(f"[A.I.R.] Stage 1 → {len(sel)} frames")

        if self.i_max == 0 or not sel:
            final = sel or list(range(0, N, max(1, N // budget)))[:budget]
            return self._save_frames(frames, final, tmp_dir)

        # ── Stage 2: iterative VLM refinement ─────────────────────────────────
        validated: set[int] = set()
        sel_set = set(sel)

        for iteration in range(self.i_max):
            if len(validated) >= budget:
                print(f"[A.I.R.] early stop iter={iteration}  validated={len(validated)}")
                break

            intervals = self._compute_potential(sims, sorted(sel_set))
            top_intervals = intervals[: self.c_intervals]
            if not top_intervals:
                break

            # representative frame per top interval (midpoint)
            candidate_indices = sorted({(a + b) // 2 for a, b, _ in top_intervals})

            cand_paths = self._save_frames(frames, candidate_indices, tmp_dir)
            scores = self._vlm_score(query, cand_paths, vlm_fn)

            newly_validated: list[int] = []
            for idx, score in zip(candidate_indices, scores):
                if score >= self.theta:
                    validated.add(idx)
                    sel_set.add(idx)
                    newly_validated.append(idx)

            print(
                f"[A.I.R.] iter {iteration + 1}/{self.i_max}  "
                f"candidates={len(candidate_indices)}  "
                f"new_validated={len(newly_validated)}  total_validated={len(validated)}"
            )

            # LDS around validated frames
            new_lds: set[int] = set()
            for anchor in newly_validated:
                new_lds |= self._lds(anchor, N, sel_set)

            if not new_lds:
                break

            # CLIP sims already computed for all N frames — just extend sel_set
            sel_set |= new_lds

        # ── final frame set ────────────────────────────────────────────────────
        # validated frames first, then remaining sel_set up to budget
        final: list[int] = sorted(validated)
        remaining_budget = budget - len(final)
        if remaining_budget > 0:
            other = sorted(sel_set - validated)
            final.extend(other[:remaining_budget])
        final = sorted(set(final))[:budget]

        # absolute fallback
        if not final:
            final = sorted(sel_set)[:budget]
        if not final:
            final = list(range(0, N, max(1, N // budget)))[:budget]

        print(f"[A.I.R.] final frames = {len(final)}")
        return self._save_frames(frames, final, tmp_dir)
