import json
import logging
import os
import random
import time
from pathlib import Path
from uuid import uuid4

from django.conf import settings


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0, maximum: int = 8) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _refcheck_scan_fps() -> float:
    return _env_float("REFCHECK_SCAN_FPS", default=3.0, minimum=0.5, maximum=6.0)


def _refcheck_dense_fps() -> float:
    return _env_float("REFCHECK_DENSE_FPS", default=15.0, minimum=2.0, maximum=30.0)


def _refcheck_dense_window_seconds() -> float:
    return _env_float("REFCHECK_DENSE_WINDOW_SECONDS", default=1.25, minimum=0.25, maximum=3.0)


def _refcheck_min_evidence_frames() -> int:
    return _env_int("REFCHECK_MIN_EVIDENCE_FRAMES", default=12, minimum=4, maximum=32)


def _refcheck_max_evidence_frames() -> int:
    return _env_int("REFCHECK_MAX_EVIDENCE_FRAMES", default=20, minimum=4, maximum=32)


def _refcheck_max_verdict_frames() -> int:
    return _env_int("REFCHECK_MAX_VERDICT_FRAMES", default=8, minimum=3, maximum=16)


def _refcheck_max_gemini_images() -> int:
    return _env_int("REFCHECK_MAX_GEMINI_IMAGES", default=24, minimum=4, maximum=48)


def _refcheck_max_candidate_frames() -> int:
    return _env_int("REFCHECK_MAX_CANDIDATE_FRAMES", default=80, minimum=20, maximum=240)


def _gemini_visual_model() -> str:
    return os.getenv("GEMINI_VISUAL_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"


def _gemini_verdict_model() -> str:
    return os.getenv("GEMINI_VERDICT_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"


def _gemini_fallback_model() -> str:
    return os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _gemini_max_retries() -> int:
    return _env_int("GEMINI_MAX_RETRIES", default=3, minimum=0, maximum=8)


def _is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None) if response is not None else None
    code_str = str(code or "").strip().lower()
    status_candidates = {status_code, response_status}
    if any(value in {408, 429, 500, 502, 503, 504} for value in status_candidates if isinstance(value, int)):
        return True
    if code_str in {"unavailable", "resource_exhausted", "deadline_exceeded"}:
        return True
    text = str(exc or "").strip().lower()
    retryable_markers = [
        "503",
        "429",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "overloaded",
        "timeout",
        "timed out",
        "deadline exceeded",
    ]
    return any(marker in text for marker in retryable_markers)


def _retry_delay_seconds(retry_index: int) -> float:
    base = min(0.6 * (2 ** retry_index), 8.0)
    return base + random.uniform(0.0, 0.35)


def _attempt_gemini_generate(
    client,
    model_name: str,
    contents: list,
    max_retries: int,
    debug: dict,
) -> tuple[str | None, Exception | None, bool]:
    from google.genai import types

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return (response.text or "").strip(), None, False
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable_gemini_error(exc)
            if retryable and attempt < max_retries:
                debug["retry_count"] = int(debug.get("retry_count", 0)) + 1
                time.sleep(_retry_delay_seconds(attempt))
                continue
            return None, exc, retryable
    return None, last_exc, _is_retryable_gemini_error(last_exc) if last_exc else False


def _generate_gemini_with_fallback(
    client,
    primary_model: str,
    fallback_model: str,
    contents: list,
    debug: dict,
) -> str:
    debug["primary_model_attempted"] = primary_model
    debug["fallback_model_attempted"] = fallback_model
    debug["retry_count"] = int(debug.get("retry_count", 0))
    debug["fallback_used"] = False
    debug["primary_model_error_message"] = ""
    debug["final_model_used"] = None

    max_retries = _gemini_max_retries()
    raw_text, primary_exc, primary_retryable = _attempt_gemini_generate(
        client=client,
        model_name=primary_model,
        contents=contents,
        max_retries=max_retries,
        debug=debug,
    )
    if raw_text is not None:
        debug["final_model_used"] = primary_model
        return raw_text

    debug["primary_model_error_message"] = str(primary_exc)[:300] if primary_exc else "Unknown primary model error."
    should_try_fallback = (
        primary_retryable
        and bool(fallback_model)
        and fallback_model.strip()
        and fallback_model.strip() != primary_model
    )
    if should_try_fallback:
        debug["fallback_used"] = True
        raw_text, fallback_exc, _ = _attempt_gemini_generate(
            client=client,
            model_name=fallback_model.strip(),
            contents=contents,
            max_retries=max_retries,
            debug=debug,
        )
        if raw_text is not None:
            debug["final_model_used"] = fallback_model.strip()
            return raw_text
        if fallback_exc:
            raise fallback_exc
    if primary_exc:
        raise primary_exc
    raise RuntimeError("Gemini request failed without a specific error.")


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    remaining_seconds = seconds - (minutes * 60)
    return f"{minutes}:{remaining_seconds:05.2f}"


def _normalize_call_type(value: str) -> str:
    allowed = {
        "blocking/charging",
        "traveling",
        "goaltending",
        "shooting foul",
        "personal foul",
        "out of bounds",
        "no-call",
        "unclear",
    }
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "unclear"


def _normalize_confidence(value: str) -> str:
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    return mapping.get((value or "").strip().lower(), "Low")


def _normalize_evidence_quality(value: str) -> str:
    candidate = (value or "").strip().title()
    if candidate in {"Good", "Limited", "Poor"}:
        return candidate
    return "Poor"


def _normalize_verdict_readiness(value: str) -> str:
    candidate = (value or "").strip().title()
    if candidate in {"Ready", "Limited", "Not Ready"}:
        return candidate
    if candidate == "Not ready":
        return "Not Ready"
    return "Not Ready"


def _normalize_sequence_interpretation(value: str) -> str:
    allowed = {"single_play", "multiple_plays", "unclear", "uncertain_continuity"}
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "unclear"


def _normalize_possible_call_type(value: str) -> str | None:
    allowed = {
        "blocking/charging",
        "personal foul",
        "traveling",
        "goaltending",
        "shooting foul",
        "out of bounds",
        "no-call",
        "unclear",
    }
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else None


def _normalize_contact_type(value: str) -> str:
    allowed = {
        "body_to_body",
        "hand_to_face",
        "arm_to_head",
        "elbow_to_head",
        "shoulder_to_head",
        "leg_to_body",
        "unclear",
    }
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "unclear"


def _normalize_contact_actor(value: str) -> str:
    allowed = {"offensive_player", "defensive_player", "both", "unclear"}
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "unclear"


def _normalize_body_area(value: str) -> str:
    allowed = {"face", "head", "neck", "torso", "arm", "leg", "unclear"}
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "unclear"


def _normalize_issue_type(value: str) -> str:
    allowed = {
        "blocking_charging",
        "personal_foul",
        "shooting_foul",
        "high_contact",
        "out_of_bounds",
        "traveling",
        "goaltending",
        "unclear",
    }
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "unclear"


def _normalize_original_call(value: str | None) -> str:
    candidate = (str(value or "")).strip()
    if candidate.lower() in {"not provided", "unknown", "none", "n/a"}:
        return ""
    return candidate


def _is_goaltending_call(value: str | None) -> bool:
    text = _normalize_original_call(value).lower()
    return "goaltend" in text or "basket interference" in text


def _normalize_role_confidence(value: str, default: str = "low") -> str:
    candidate = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {"low", "medium", "high", "low_to_medium", "medium_to_high"}
    return candidate if candidate in allowed else default


def _normalize_secondary_issues(payload_items: list) -> list[dict]:
    if not isinstance(payload_items, list):
        return []
    normalized = []
    for item in payload_items[:6]:
        if not isinstance(item, dict):
            continue
        issue = _complete_sentence(str(item.get("issue", "")).strip(), max_len=220)
        reason = _complete_sentence(str(item.get("reason", "")).strip(), max_len=500)
        if not issue and not reason:
            continue
        normalized.append(
            {
                "issue": issue or "Possible secondary contact/context issue.",
                "confidence": _normalize_role_confidence(item.get("confidence", "low_to_medium"), default="low_to_medium"),
                "reason": reason or "Visible contact may exist but is secondary to the selected call.",
            }
        )
    return normalized


def _normalize_contact_direction(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    cleaned = {
        "contact_initiator": str(payload.get("contact_initiator", "")).strip()[:160],
        "contact_recipient": str(payload.get("contact_recipient", "")).strip()[:160],
        "contact_location": str(payload.get("contact_location", "")).strip()[:180],
        "contact_context": str(payload.get("contact_context", "")).strip()[:320],
        "contact_effect": str(payload.get("contact_effect", "")).strip()[:280],
        "contact_confidence": _normalize_role_confidence(payload.get("contact_confidence", "low"), default="low"),
    }
    return {key: value for key, value in cleaned.items() if value}


def _normalize_event_tier(value: str) -> str:
    candidate = " ".join((value or "").strip().split()).upper()
    if candidate in {"A", "TIER A"}:
        return "Tier A"
    if candidate in {"B", "TIER B"}:
        return "Tier B"
    if candidate in {"C", "TIER C"}:
        return "Tier C"
    if candidate in {"D", "TIER D"}:
        return "Tier D"
    return "Tier D"


def _event_tier_rank(tier: str) -> int:
    return {"Tier A": 4, "Tier B": 3, "Tier C": 2, "Tier D": 1}.get(_normalize_event_tier(tier), 1)


def _timestamp_to_seconds(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        if ":" not in text:
            return float(text)
        minutes, seconds = text.split(":", 1)
        return (float(minutes) * 60.0) + float(seconds)
    except (TypeError, ValueError):
        return 0.0


def _normalize_why_relevant(value: str) -> str:
    allowed = {
        "possible contact",
        "defender position",
        "ball release",
        "referee signal",
        "boundary",
        "other",
    }
    candidate = (value or "").strip().lower()
    return candidate if candidate in allowed else "other"


def _complete_sentence(value: str, max_len: int = 500) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        return ""
    if len(text) > max_len:
        boundary = max(text.rfind(".", 0, max_len), text.rfind("!", 0, max_len), text.rfind("?", 0, max_len))
        if boundary >= max_len * 0.45:
            text = text[: boundary + 1]
        else:
            text = text[:max_len].rstrip()
    if text[-1] not in ".!?":
        text += "."
    return text


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _extract_json_object(text: str) -> str | None:
    source = _strip_code_fences(text)
    if source.startswith("{") and source.endswith("}"):
        return source
    start = source.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start : index + 1]
        start = source.find("{", start + 1)
    return None


def _safe_excerpt(text: str, max_len: int = 500) -> str:
    if not text:
        return ""
    return text.strip().replace("\n", " ")[:max_len]


def _safe_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _uniform_fallback_indices(total_frames: int, desired: int = 8) -> list[int]:
    target = min(max(1, desired), max(1, total_frames))
    if target == 1:
        return [max(0, total_frames // 2)]
    indices = []
    for i in range(target):
        idx = int(round(i * (total_frames - 1) / (target - 1)))
        indices.append(max(0, min(total_frames - 1, idx)))
    deduped = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def _dedupe_sorted_indices(indices: list[int], total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    seen = set()
    deduped = []
    for idx in indices:
        bounded = max(0, min(total_frames - 1, int(idx)))
        if bounded in seen:
            continue
        seen.add(bounded)
        deduped.append(bounded)
    return sorted(deduped)


def _fractional_frame_indices(total_frames: int, fractions: list[float]) -> list[int]:
    if total_frames <= 0:
        return []
    return [
        int(round((total_frames - 1) * max(0.0, min(1.0, fraction))))
        for fraction in fractions
    ]


def _candidate_indices(total_frames: int, fps: float, duration: float) -> list[int]:
    if total_frames <= 0:
        return []
    scan_fps = _refcheck_scan_fps() if duration <= 20 else min(_refcheck_scan_fps(), 2.0)
    step = max(1, int(round(fps / scan_fps))) if fps > 0 else 1
    indices = list(range(0, total_frames, step))
    indices.extend(
        _fractional_frame_indices(
            total_frames,
            [0.0, 0.03, 0.08, 0.15, 0.25, 0.375, 0.50, 0.625, 0.75, 0.85, 0.92, 0.97, 1.0],
        )
    )
    max_candidates = _refcheck_max_candidate_frames()
    indices = _dedupe_sorted_indices(indices, total_frames)
    if len(indices) > max_candidates:
        protected = set(
            _fractional_frame_indices(total_frames, [0.0, 0.03, 0.08, 0.50, 0.92, 0.97, 1.0])
        )
        remaining_budget = max(0, max_candidates - len(protected))
        sampled = []
        if remaining_budget > 0:
            for i in range(remaining_budget):
                pos = int(round(i * (len(indices) - 1) / max(1, remaining_budget - 1)))
                sampled.append(indices[pos])
        indices = _dedupe_sorted_indices([*protected, *sampled], total_frames)
        if len(indices) > max_candidates:
            indices = indices[:max_candidates]
    return indices


def _collect_candidates(capture, indices: list[int], fps: float):
    import cv2

    candidates = []
    prev_small_gray = None
    prev_hist = None

    for i, frame_idx in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        max_width = 1024
        original_h, original_w = frame.shape[:2]
        if original_w > max_width:
            scale = max_width / float(original_w)
            resized_h = int(original_h * scale)
            frame = cv2.resize(frame, (max_width, resized_h), interpolation=cv2.INTER_AREA)

        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        motion = 0.0
        lower_motion = 0.0
        motion_regions = 0
        if prev_small_gray is not None:
            diff = cv2.absdiff(gray, prev_small_gray)
            motion = float(diff.mean() / 255.0)
            lower_motion = float(diff[int(diff.shape[0] * 0.55):, :].mean() / 255.0)
            _, diff_mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_regions = sum(1 for contour in contours if cv2.contourArea(contour) >= 20)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        court_mask = cv2.inRange(hsv, (8, 30, 60), (35, 220, 255))
        court_ratio = float(court_mask.mean() / 255.0)

        h, w = gray.shape
        c0, c1 = int(w * 0.2), int(w * 0.8)
        r0, r1 = int(h * 0.2), int(h * 0.8)
        center = gray[r0:r1, c0:c1]
        edges = cv2.Canny(center, 80, 160)
        center_edge_density = float(edges.mean() / 255.0)
        relevance = min(1.0, (court_ratio * 1.3) + (center_edge_density * 1.6))
        lower = gray[int(h * 0.55):, :]
        lower_edges = cv2.Canny(lower, 80, 160)
        lower_edge_density = float(lower_edges.mean() / 255.0)
        lower_body_cluster_score = min(1.0, lower_edge_density * 3.5)
        floor_activity_score = min(1.0, (lower_motion * 5.0) + (lower_edge_density * 2.0))
        player_heap_score = min(1.0, (lower_edge_density * 2.5) + (min(motion_regions, 8) / 12.0))

        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        duplicate_similarity = 0.0
        if prev_hist is not None:
            duplicate_similarity = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL))
        duplicate_similarity = max(-1.0, min(1.0, duplicate_similarity))
        timestamp_seconds = (frame_idx / fps) if fps > 0 else 0.0
        frame_context = {
            "lower_body_cluster_score": lower_body_cluster_score,
            "floor_activity_score": floor_activity_score,
            "player_heap_score": player_heap_score,
            "motion_peak_group_id": None,
        }
        candidates.append(
            {
                "candidate_id": i,
                "frame_idx": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "timestamp": _format_timestamp(timestamp_seconds),
                "frame": frame,
                "motion": motion,
                "sharpness": sharpness,
                "duplicate_similarity": duplicate_similarity,
                "relevance": relevance,
                "score": 0.0,
                "lower_body_cluster_score": lower_body_cluster_score,
                "floor_activity_score": floor_activity_score,
                "player_heap_score": player_heap_score,
                "frame_context": frame_context,
                "selection_reason": "selected for temporal coverage",
            }
        )

        prev_small_gray = gray
        prev_hist = hist
    return candidates


def _score_candidates(candidates: list[dict]) -> None:
    if not candidates:
        return
    motions = [float(c.get("motion", 0.0)) for c in candidates]
    sharpnesses = [float(c.get("sharpness", 0.0)) for c in candidates]
    relevances = [float(c.get("relevance", 0.0)) for c in candidates]
    floor_scores = [float(c.get("floor_activity_score", 0.0)) for c in candidates]
    heap_scores = [float(c.get("player_heap_score", 0.0)) for c in candidates]

    motion_norm = _safe_minmax(motions)
    sharp_norm = _safe_minmax(sharpnesses)
    relevance_norm = _safe_minmax(relevances)
    floor_norm = _safe_minmax(floor_scores)
    heap_norm = _safe_minmax(heap_scores)

    for i, candidate in enumerate(candidates):
        duplicate_penalty = max(0.0, float(candidate.get("duplicate_similarity", 0.0)))
        uniqueness = 1.0 - duplicate_penalty
        score = (
            (0.28 * motion_norm[i])
            + (0.24 * sharp_norm[i])
            + (0.23 * relevance_norm[i])
            + (0.12 * floor_norm[i])
            + (0.06 * heap_norm[i])
            + (0.07 * uniqueness)
        )
        reason = str(candidate.get("selection_reason", ""))
        if reason in {"clip start context", "clip end context"}:
            score += 0.04
        candidate["score"] = float(min(1.0, score))
        candidate["motion_norm"] = float(motion_norm[i])


def _selection_phase(reason: str) -> str:
    text = str(reason or "").lower()
    if "clip start" in text:
        return "clip_start_context"
    if "clip end" in text:
        return "clip_end_context"
    if "pre-contact" in text or "before contact" in text:
        return "pre_contact"
    if any(
        marker in text
        for marker in ["likely contact", "high-motion peak", "contact moment", "high contact", "contact point"]
    ):
        return "likely_contact"
    if "post-contact" in text or "after contact" in text:
        return "post_contact"
    if "aftermath" in text or "player-on-floor" in text:
        return "aftermath"
    return "temporal_coverage"


def _tag_boundary_candidate_reasons(candidates: list[dict], total_frames: int) -> None:
    if not candidates or total_frames <= 0:
        return
    start_cutoff = max(0, int(round((total_frames - 1) * 0.08)))
    end_cutoff = min(total_frames - 1, int(round((total_frames - 1) * 0.92)))
    first = min(candidates, key=lambda item: int(item.get("frame_idx", 0)))
    last = max(candidates, key=lambda item: int(item.get("frame_idx", 0)))
    for candidate in candidates:
        frame_idx = int(candidate.get("frame_idx", 0))
        if frame_idx == int(first.get("frame_idx", 0)) or frame_idx <= start_cutoff:
            candidate["selection_reason"] = "clip start context"
        elif frame_idx == int(last.get("frame_idx", 0)) or frame_idx >= end_cutoff:
            candidate["selection_reason"] = "clip end context"
        elif candidate.get("selection_reason") == "selected for temporal coverage":
            candidate["selection_reason"] = "temporal coverage"
        candidate["phase_bucket"] = _selection_phase(candidate.get("selection_reason", ""))


def _phase_bucket_counts(candidates: list[dict]) -> dict:
    counts = {
        "clip_start_context": 0,
        "pre_contact": 0,
        "likely_contact": 0,
        "post_contact": 0,
        "aftermath": 0,
        "clip_end_context": 0,
        "temporal_coverage": 0,
    }
    for candidate in candidates:
        phase = candidate.get("phase_bucket") or _selection_phase(candidate.get("selection_reason", ""))
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _rank_motion_peaks(
    candidates: list[dict],
    fps: float,
    min_separation_seconds: float = 1.0,
    max_peaks: int = 5,
) -> list[dict]:
    if not candidates:
        return []
    _score_candidates(candidates)

    ordered = sorted(candidates, key=lambda item: int(item.get("frame_idx", 0)))
    local_peaks = []
    for i, candidate in enumerate(ordered):
        motion = float(candidate.get("motion", 0.0))
        prev_motion = float(ordered[i - 1].get("motion", 0.0)) if i > 0 else -1.0
        next_motion = float(ordered[i + 1].get("motion", 0.0)) if i < len(ordered) - 1 else -1.0
        if motion > 0 and motion >= prev_motion and motion >= next_motion:
            local_peaks.append(candidate)

    ranked = sorted(
        local_peaks or ordered,
        key=lambda item: (
            float(item.get("motion_norm", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    min_distance = max(1, int(round(min_separation_seconds * fps))) if fps > 0 else 1
    accepted = []
    for peak in ranked:
        frame_idx = int(peak.get("frame_idx", 0))
        if any(abs(frame_idx - int(existing.get("frame_idx", 0))) < min_distance for existing in accepted):
            continue
        accepted.append(peak)
        if len(accepted) >= max_peaks:
            break
    return accepted


def _dense_indices_for_motion_peaks(
    peaks: list[dict],
    total_frames: int,
    fps: float,
    dense_fps: float,
    window_seconds: float,
) -> tuple[list[int], list[dict]]:
    if not peaks or fps <= 0 or total_frames <= 0:
        return [], []
    step = max(1, int(round(fps / dense_fps)))
    radius = max(1, int(round(window_seconds * fps)))
    indices: set[int] = set()
    peak_debug = []
    for group_id, peak in enumerate(peaks, start=1):
        center = int(peak.get("frame_idx", 0))
        start = max(0, center - radius)
        end = min(total_frames - 1, center + radius)
        for frame_idx in range(start, end + 1, step):
            indices.add(frame_idx)
        indices.add(center)
        peak["motion_peak_group_id"] = group_id
        peak_debug.append(
            {
                "group_id": group_id,
                "timestamp": peak.get("timestamp", ""),
                "frame_idx": center,
                "motion": round(float(peak.get("motion", 0.0)), 5),
                "motion_norm": round(float(peak.get("motion_norm", 0.0)), 3),
                "score": round(float(peak.get("score", 0.0)), 3),
            }
        )
    return sorted(indices), peak_debug


def _tag_dense_candidate_reasons(candidates: list[dict], peaks: list[dict], fps: float, window_seconds: float) -> None:
    if not candidates or not peaks or fps <= 0:
        return
    for candidate in candidates:
        frame_idx = int(candidate.get("frame_idx", 0))
        nearest = min(peaks, key=lambda peak: abs(frame_idx - int(peak.get("frame_idx", 0))))
        center = int(nearest.get("frame_idx", 0))
        delta_seconds = (frame_idx - center) / fps
        group_id = int(nearest.get("motion_peak_group_id") or 0)
        candidate["motion_peak_group_id"] = group_id
        frame_context = candidate.setdefault("frame_context", {})
        frame_context["motion_peak_group_id"] = group_id
        if delta_seconds < -0.35:
            reason = "pre-contact context"
        elif delta_seconds <= 0.35:
            reason = "likely contact moment"
        elif delta_seconds <= max(0.75, window_seconds * 0.65):
            reason = "post-contact context"
        else:
            reason = "aftermath context"
        if abs(delta_seconds) <= 0.12:
            reason = "likely contact moment"
        if candidate.get("floor_activity_score", 0.0) >= 0.28 or candidate.get("player_heap_score", 0.0) >= 0.45:
            reason = "player-on-floor context" if delta_seconds > 0.35 else reason
        candidate["selection_reason"] = reason
        candidate["phase_bucket"] = _selection_phase(reason)


def _is_chaotic_play_suspected(candidates: list[dict], peaks: list[dict]) -> bool:
    if not candidates:
        return False
    high_floor_count = sum(
        1
        for candidate in candidates
        if candidate.get("floor_activity_score", 0.0) >= 0.30 or candidate.get("player_heap_score", 0.0) >= 0.50
    )
    high_motion_count = sum(1 for candidate in candidates if candidate.get("motion_norm", 0.0) >= 0.65)
    aftermath_low_motion = False
    for peak in peaks[:3]:
        peak_time = float(peak.get("timestamp_seconds", 0.0))
        later = [
            candidate for candidate in candidates
            if 0.25 <= float(candidate.get("timestamp_seconds", 0.0)) - peak_time <= 2.0
        ]
        if any(
            candidate.get("motion_norm", 0.0) <= 0.35 and candidate.get("floor_activity_score", 0.0) >= 0.24
            for candidate in later
        ):
            aftermath_low_motion = True
            break
    return high_floor_count >= 3 or (high_motion_count >= 3 and aftermath_low_motion)


def _cap_candidate_pool(candidates: list[dict], max_candidates: int) -> list[dict]:
    if len(candidates) <= max_candidates:
        return sorted(candidates, key=lambda item: item.get("timestamp_seconds", 0.0))
    _score_candidates(candidates)
    action = [
        item for item in candidates
        if (item.get("phase_bucket") or _selection_phase(item.get("selection_reason", "")))
        in {"pre_contact", "likely_contact", "post_contact", "aftermath"}
    ]
    action_ids = {id(item) for item in action}
    context = [item for item in candidates if id(item) not in action_ids]
    keep: list[dict] = []
    seen: set[int] = set()

    for item in sorted(action, key=lambda c: c.get("score", 0.0), reverse=True):
        frame_idx = int(item.get("frame_idx", -1))
        if frame_idx in seen:
            continue
        keep.append(item)
        seen.add(frame_idx)
        if len(keep) >= int(max_candidates * 0.75):
            break

    for item in sorted(context, key=lambda c: (c.get("score", 0.0), -c.get("timestamp_seconds", 0.0)), reverse=True):
        if len(keep) >= max_candidates:
            break
        frame_idx = int(item.get("frame_idx", -1))
        if frame_idx in seen:
            continue
        keep.append(item)
        seen.add(frame_idx)
    return sorted(keep, key=lambda item: item.get("timestamp_seconds", 0.0))


def _reindex_candidates(candidates: list[dict]) -> None:
    for index, candidate in enumerate(sorted(candidates, key=lambda item: item.get("timestamp_seconds", 0.0))):
        candidate["candidate_id"] = index



def _merge_candidates(primary: list[dict], dense: list[dict]) -> list[dict]:
    by_frame: dict[int, dict] = {}
    for candidate in [*primary, *dense]:
        frame_idx = int(candidate.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        existing = by_frame.get(frame_idx)
        if existing is None or (
            candidate.get("score", 0.0),
            candidate.get("sharpness", 0.0),
        ) > (
            existing.get("score", 0.0),
            existing.get("sharpness", 0.0),
        ):
            by_frame[frame_idx] = candidate
    merged = sorted(by_frame.values(), key=lambda item: item.get("timestamp_seconds", 0.0))
    for i, candidate in enumerate(merged):
        candidate["candidate_id"] = i
    return merged


def _candidate_selection_weight(candidate: dict) -> float:
    reason = str(candidate.get("selection_reason", ""))
    weight = float(candidate.get("score", 0.0))
    if "likely contact" in reason or "contact moment" in reason:
        weight += 0.45
    elif "pre-contact" in reason:
        weight += 0.32
    elif "post-contact" in reason or "aftermath" in reason:
        weight += 0.28
    elif "player-on-floor" in reason or "loose-ball" in reason:
        weight += 0.24
    elif "clip start" in reason or "clip end" in reason:
        weight += 0.18
    return weight


def _select_evidence_candidates(
    candidates: list[dict],
    target_min: int = 8,
    target_max: int = 12,
    chaotic_play_suspected: bool = False,
):
    if not candidates:
        return []

    _score_candidates(candidates)
    for candidate in candidates:
        candidate["phase_bucket"] = candidate.get("phase_bucket") or _selection_phase(
            candidate.get("selection_reason", "")
        )

    selected = []
    selected_ids = set()
    min_spacing = 3

    def add_candidate(item: dict, reason: str | None = None, force: bool = False) -> bool:
        if len(selected) >= target_max or item.get("candidate_id") in selected_ids:
            return False
        if not force and len(selected) >= target_min:
            too_close = any(
                abs(int(item.get("frame_idx", 0)) - int(existing.get("frame_idx", 0))) <= min_spacing
                for existing in selected
            )
            if too_close:
                return False
        if reason:
            item["selection_reason"] = reason
        item["phase_bucket"] = _selection_phase(item.get("selection_reason", ""))
        selected.append(item)
        selected_ids.add(item["candidate_id"])
        return True

    def best_for_phase(phase: str, prefer_earliest: bool = False, prefer_latest: bool = False) -> dict | None:
        phase_candidates = [
            item for item in candidates
            if (item.get("phase_bucket") or _selection_phase(item.get("selection_reason", ""))) == phase
        ]
        if not phase_candidates:
            return None
        if prefer_earliest:
            return min(phase_candidates, key=lambda c: (c.get("timestamp_seconds", 0.0), -c.get("score", 0.0)))
        if prefer_latest:
            return max(phase_candidates, key=lambda c: (c.get("timestamp_seconds", 0.0), c.get("score", 0.0)))
        return max(phase_candidates, key=_candidate_selection_weight)

    start = best_for_phase("clip_start_context", prefer_earliest=True)
    if start:
        add_candidate(start, "clip start context", force=True)
    end = best_for_phase("clip_end_context", prefer_latest=True)
    if end:
        add_candidate(end, "clip end context", force=True)

    for phase in ["pre_contact", "likely_contact", "post_contact"]:
        item = best_for_phase(phase)
        if item:
            add_candidate(item, force=True)

    if chaotic_play_suspected:
        item = best_for_phase("aftermath")
        if item:
            add_candidate(item, force=True)

    for phase in ["temporal_coverage", "aftermath"]:
        item = best_for_phase(phase)
        if item:
            add_candidate(item)

    for item in sorted(candidates, key=_candidate_selection_weight, reverse=True):
        if len(selected) >= target_max:
            break
        if item.get("candidate_id") in selected_ids:
            continue
        if item["selection_reason"] == "selected for temporal coverage":
            if item["motion_norm"] >= 0.55:
                item["selection_reason"] = "likely contact moment"
            elif item["motion_norm"] <= 0.20:
                item["selection_reason"] = "temporal coverage"
            else:
                item["selection_reason"] = "temporal coverage"
        add_candidate(item)

    if len(selected) < target_min:
        for item in sorted(candidates, key=lambda c: c["timestamp_seconds"]):
            if len(selected) >= target_min:
                break
            if item.get("candidate_id") in selected_ids:
                continue
            add_candidate(item, force=True)

    return sorted(selected[:target_max], key=lambda c: c["timestamp_seconds"])


def _clean_for_session(result: dict) -> None:
    for frame in result["evidence_frames"]:
        frame.pop("local_path", None)


def _default_recommended_frames(evidence_frames: list[dict]) -> list[dict]:
    if not evidence_frames:
        return []
    priority = []
    for frame in evidence_frames:
        reason = frame.get("selection_reason", "")
        weight = 0
        if "likely contact" in reason or "contact moment" in reason or "high motion" in reason:
            weight = 5
        elif "pre-contact" in reason:
            weight = 4
        elif "post-contact" in reason:
            weight = 3
        elif "aftermath" in reason or "player-on-floor" in reason:
            weight = 2
        elif "clip start" in reason or "clip end" in reason or "temporal coverage" in reason:
            weight = 1
        priority.append((weight, frame))
    priority.sort(key=lambda item: (item[0], item[1].get("score", 0.0)), reverse=True)
    top = sorted([item[1] for item in priority[:_refcheck_max_verdict_frames()]], key=lambda frame: frame.get("timestamp", ""))
    return [
        {
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "reason": _complete_sentence(
                f"Clear visual frame: {frame.get('selection_reason', 'selected for temporal coverage')}"
            ),
        }
        for frame in top
    ]


_VISUAL_PROMPT_INSTRUCTIONS = [
    "# ROLE",
    "You are Agent 1: Visual Analyst in a multi-agent officiating-review pipeline.",
    "Your only job is to produce a detailed, neutral, frame-grounded description of exactly what is visible in the selected video frames.",
    "You are not a referee, not a rules analyst, and not the final decision maker. A separate downstream Verdict Agent will handle rules and the final call.",
    "You only see a small number of selected still frames, not the full continuous video. Treat anything between frames as unseen.",
    "",
    "# REQUIRED TWO-PASS WORKFLOW",
    "Pass A - Triage: Before producing detailed analysis, identify all potentially relevant officiating events visible in the provided frames, assign each one a severity tier, and choose the primary_event.",
    "Severity tiers:",
    "Tier A — Most serious. Contact to head, face, or neck. Deliberate strike with hand, elbow, knee, or foot. Player on the floor being stepped on. Anything that would warrant a flagrant or technical foul.",
    "Tier B — Serious. Hard body contact during a play, late hit after a whistle, push that sends a player to the floor, blocking/charging on a drive, illegal screens with significant impact.",
    "Tier C — Incidental. Minor pushing, jostling for position, light contact during rebounds, hand-checking, brushing.",
    "Tier D — Non-event. Routine basketball action, no contact, regular movement.",
    "Primary event selection rule: The primary event is the highest-tier event in the clip. If multiple events are at the same tier, the primary event is the one that most directly affects the play's outcome - typically the latest event, since referees usually call what defined the action. Earlier lower-tier events are CONTEXT, not the call.",
    "Pass B - Deep analysis: Only after choosing primary_event, produce the rest of the structured analysis focused on the primary_event. Non-primary events may be mentioned briefly as context in video_summary, but do not analyze them in detail in issue_cards or contact_points.",
    "issue_cards, contact_points, possible_high_contact, high_contact_* fields, possible_critical_moments, and recommended_frames_for_verdict_agent must prioritize the primary_event rather than averaging all visible events.",
    "",
    "# WHAT TO DESCRIBE (BE EXHAUSTIVE, DO NOT LEAVE OUT DETAILS)",
    "Describe every visually relevant element. Aim for a thorough, evidence-grade description rather than a short summary.",
    "For each frame and across the sequence, cover at minimum:",
    "- Setting and context: court or field type if visible, visible court markings (three-point line, free-throw line, restricted area arc, sideline, baseline, midcourt line, key/paint), scoreboard or clock overlays, broadcast graphics, on-screen text, score, time remaining, and any visible 'REPLAY' or angle labels.",
    "- Camera: framing (wide, medium, tight), apparent angle (baseline, sideline, overhead, behind-the-basket), zoom level, and whether the angle changes between frames.",
    "- Players involved: jersey color, jersey number if legible, approximate role (ball handler, defender, shooter, screener, off-ball), and stance. Refer to players consistently across frames (e.g., 'white #23', 'dark-jersey defender').",
    "- Body positions and biomechanics: feet placement (set, sliding, airborne, on heels, on toes), torso lean, hip orientation, arm/hand position, head direction, point of balance, and whether the player appears stationary or moving.",
    "- Ball state: who possesses it, height (floor, waist, chest, above head), whether it is being dribbled, gathered, released, in flight, deflected, or loose; visible spin or trajectory if discernible.",
    "- Contact: where on the body contact appears to occur (chest-to-chest, hip, shoulder, arm, hand, leg, head), which player initiates visible contact relative to the frame shown, and the visible reaction (recoil, fall, no reaction).",
    "- High contact: explicitly inspect the face, head, and neck area of every involved player. Look for hand, arm, elbow, forearm, or shoulder contact near the face/head/neck. Chest/body contact must not hide or replace a separate possible high-contact issue.",
    "- If a hand/arm/elbow/shoulder is close to the face/head/neck but actual contact is unclear, mark possible_high_contact=true with Low confidence and describe the uncertainty.",
    "- Spatial relationships: distance between players, who is closer to the basket/boundary, feet relative to lines (inside/outside the restricted area, on or beyond the three-point line, in/out of bounds).",
    "- Officials: position of any visible referee, whistle, arm signal, or pointing direction. Only describe what is visibly shown.",
    "- Sequence/continuity: how positions, possession, and contact change from one selected frame to the next, and whether 'before / during / after' phases of an event are present in the selection.",
    "Use precise visual nouns and verbs. Prefer 'the white-jersey player extends their right arm into the chest of the dark-jersey player' over 'a foul occurs'.",
    "If a detail is partially visible, say so explicitly (for example, 'the defender's left foot is cropped out of frame, so its placement relative to the restricted-area arc is not determinable').",
    "If a detail is not visible at all, do not invent it; record it under missing_evidence.",
    "",
    "# NEUTRALITY (HARD RULES)",
    "Use neutral visual language only. Never label actions as legal, illegal, correct, incorrect, foul, violation, travel, charge, block, or no-call unless such wording is literally rendered as text/overlay in the image itself.",
    "Do not decide Fair Call or Bad Call. Do not say whether the official was correct or incorrect. Do not cite rulebooks or apply rules reasoning.",
    "Do not infer intent ('he meant to', 'tried to'), causation ('because of'), exact timing in seconds between frames, or what happened during gaps you cannot see.",
    "If contact is visible, describe only the visible body positions and contact appearance. Do not assign responsibility, blame, or fault.",
    "",
    "# SEQUENCE CONTINUITY",
    "Distinguish between a single continuous play and clearly separate plays only when the selected frames visibly support that distinction.",
    "Default assumption is single_play unless the frames clearly show a different possession, different players, or a different court location.",
    "If continuity is uncertain, set sequence_interpretation to 'uncertain_continuity' and explain the specific visual uncertainty in sequence_interpretation_reason.",
    "",
    "# SELECTED CALL ANCHOR (CRITICAL)",
    "The user-provided original call is the review anchor for primary_issue.",
    "If original_call is Goaltending, set primary_issue='goaltending' and prioritize ball/rim/backboard/cylinder evidence over generic body contact.",
    "For Goaltending-focused clips, explicitly inspect: ball trajectory (upward/downward), ball position relative to cylinder, whether ball contacted backboard before defender contact, and whether exact defender-ball touch frame is visible.",
    "If those goaltending elements are not clearly visible, keep verdict-readiness low and list the missing ball-evidence items under missing_evidence.",
    "Do not let a secondary body-contact observation replace goaltending as the primary_issue when original_call is Goaltending.",
    "",
    "# ROLE + CONTACT DIRECTION MODEL",
    "Infer team roles separately from jersey colors. Do not assume the nearest player is the shooter.",
    "Use possession, pass direction, ball control, and finish attempt context to identify offense and defense. If unclear, output unknown with low confidence and explain why.",
    "Model contact direction explicitly: who initiated contact, who received it, where contact occurred, context (on-ball/off-ball/airborne contest), and likely effect if visible.",
    "If contact direction is ambiguous, state uncertainty explicitly instead of forcing a directional claim.",
    "",
    "# CALL TYPE FIELDS",
    "Do not infer visible_call_type from player actions. Player movement alone never establishes a call type.",
    "Set visible_call_type to 'unclear' unless the call type is explicitly visible as on-screen text/overlay/graphic, or has been provided by the user as the original call.",
    "If visible_call_type uses the user-provided original call, visible_call_type_reason must explicitly state that the value is user-provided and is not a visual or rules conclusion drawn by you.",
    "visible_call_type_reason must either (a) cite the exact visible basis (e.g., 'overlay text reads BLOCK on frame_004') or (b) state that Agent 1 is not making a call classification.",
    "possible_call_types may list visual categories a later Verdict Agent might consider, but this is suggestive, not a final decision.",
    "issue_cards must list plausible officiating issues for the primary_event. Do not create detailed issue cards for lower-tier context events unless they are inseparable from the primary_event.",
    "For each issue_card, write rag_query as a concise basketball rule search query based on the visual evidence, not a verdict.",
    "officiating_issue_summary must be a neutral, visually grounded summary of what a future Verdict Agent may need to examine. It is not a rules conclusion and must not contain a verdict.",
    "",
    "# QUALITY AND READINESS FIELDS",
    "visual_frame_quality rates image clarity and usefulness only (lighting, focus, motion blur, occlusion, resolution, framing). It does not rate the play itself.",
    "verdict_readiness rates whether a future Verdict Agent has enough visual continuity (before / during / after the key moment) to reason about the call.",
    "can_reason_about_call indicates only whether selected frames contain enough visual continuity for a later Verdict Agent to reason. It is not your decision on the call.",
    "Set can_reason_about_call to false if any of the following hold: defender feet are not clearly visible before contact, restricted-area or boundary context is unclear, the exact moment of contact is not continuously captured, or referee signal or original call is not visible and not provided by the user.",
    "If can_reason_about_call is false, missing_evidence must enumerate exactly what is missing in concrete visual terms (e.g., 'no frame shows the defender's feet at the moment of contact').",
    "limitations should list visual constraints that affect interpretation even when readiness is otherwise acceptable (e.g., 'partial occlusion of the ball-handler's lower body in frame_003').",
    "",
    "# FIELD-BY-FIELD CONTRACT (READ CAREFULLY)",
    "video_summary RULES:",
    "- Write video_summary as a single continuous neutral paragraph of plain descriptive prose, multiple sentences, in temporal order, as if narrating continuous footage.",
    "- DO NOT mention frame_id, frame numbers, the word 'frame', timestamps, or any reference to the selection, scoring, or recommendation process.",
    "- DO NOT mention Agent 1, Agent 2, the Verdict Agent, can_reason_about_call, evidence quality, or any pipeline/metadata field.",
    "- DO NOT include rules reasoning, verdicts, fairness judgments, or call-type labels.",
    "- video_summary must remain self-contained and reusable as an embedding/search query. Avoid pipeline jargon. Describe only what is visibly happening.",
    "- Cover setting, players, ball state, contact if any, and how the action progresses, but as flowing prose only.",
    "Frame-bound fields (all_events_detected, primary_event, key_events, possible_critical_moments, recommended_frames_for_verdict_agent) are the ONLY fields that may cite frame_id and timestamps.",
    "- Each entry in those arrays MUST include the frame_id and timestamp it refers to.",
    "- Each key_events[].description must be a complete declarative sentence grounded in the cited frame.",
    "- Each possible_critical_moments[].description must be a complete sentence describing what is visible in the cited frame.",
    "- Recommend frames based only on visual clarity, sequence relevance, and whether they show before/during/after positions of the key moment.",
    "Quality/readiness fields (visual_frame_quality, verdict_readiness, can_reason_about_call, sequence_interpretation, sequence_interpretation_reason, missing_evidence, limitations) live as their own structured values. Never narrate them inside video_summary.",
    "officiating_issue_summary may reference what a downstream reviewer should examine, but must remain neutral and visually grounded; it is NOT a verdict.",
    "",
    "# EXAMPLES OF GOOD VS FORBIDDEN video_summary",
    "GOOD: 'A white-jersey ball-handler drives along the right baseline; a dark-jersey defender slides into their path with both feet set, and the two make chest-to-chest contact, after which the ball-handler falls to the floor.'",
    "FORBIDDEN: 'In frame_003 at 0:02.50 the defender is set; this is recommended for the Verdict Agent.' (mentions frame_id, timestamp, and pipeline jargon)",
    "FORBIDDEN: 'This appears to be a charging foul because the defender was set first.' (rules reasoning / verdict)",
    "",
    "# SEVERITY TRIAGE EXAMPLES",
    "Example 1: Clip shows two players bumping shoulders at 0:01, then at 0:04 player A's open hand makes clear contact with player B's face. Output: all_events_detected lists both. primary_event is the 0:04 face contact (Tier A). The shoulder bump is mentioned in video_summary as preliminary context but issue_cards focus on the face contact only.",
    "Example 2: Clip shows a drive to the basket where a defender steps in late and contact occurs at 0:03. No other significant events. Output: all_events_detected has one entry. primary_event is the 0:03 block/charge (Tier B). issue_cards analyze blocking_charging as before.",
    "Example 3 anti-pattern: WRONG output for a clip with light pushing then a strike: 'Players were pushing each other and there was contact.' This averages events of different severity. RIGHT output: identify the strike as primary event Tier A; mention pushing only as preliminary context in video_summary.",
    "",
    "# OUTPUT REQUIREMENTS",
    "Return valid JSON only, with no prose outside the JSON, no markdown fences, and exactly this schema:",
    "{",
    '  "agent": "visual_analyst",',
    '  "all_events_detected": [{"tier":"Tier A|Tier B|Tier C|Tier D","timestamp":"M:SS.ss","frame_ids":["frame_00x"],"brief_description":"string"}],',
    '  "primary_event": {"tier":"Tier A|Tier B|Tier C|Tier D","timestamp":"M:SS.ss","frame_ids":["frame_00x"],"description":"string","why_this_is_primary":"string"},',
    '  "video_summary": "string",',
    '  "key_events": [{"timestamp": "M:SS.ss", "frame_id": "frame_00x", "description": "string"}],',
    '  "sequence_interpretation": "single_play|multiple_plays|unclear|uncertain_continuity",',
    '  "sequence_interpretation_reason": "string",',
    '  "primary_issue": "goaltending|blocking/charging|traveling|shooting foul|personal foul|out of bounds|no-call|unclear",',
    '  "primary_reason": "string",',
    '  "offense_team_color": "string",',
    '  "defense_team_color": "string",',
    '  "shooter_or_finisher": "string",',
    '  "primary_contesting_defender": "string",',
    '  "role_confidence": "low|medium|high|low_to_medium|medium_to_high",',
    '  "role_uncertainty_reason": "string",',
    '  "contact_direction_assessment": {"contact_initiator":"string","contact_recipient":"string","contact_location":"string","contact_context":"string","contact_effect":"string","contact_confidence":"low|medium|high|low_to_medium|medium_to_high"},',
    '  "secondary_issues": [{"issue":"string","confidence":"low|medium|high|low_to_medium|medium_to_high","reason":"string"}],',
    '  "possible_critical_moments": [',
    '    {"timestamp": "M:SS.ss", "frame_id": "frame_00x", "description": "string", "why_relevant": "possible contact|defender position|ball release|referee signal|boundary|other"}',
    "  ],",
    '  "visible_call_type": "blocking/charging|traveling|goaltending|shooting foul|personal foul|out of bounds|no-call|unclear",',
    '  "visible_call_type_reason": "string",',
    '  "possible_call_types": ["blocking/charging", "personal foul"],',
    '  "contact_points": [',
    '    {"frame_id":"frame_00x","timestamp":"M:SS.ss","contact_type":"body_to_body|hand_to_face|arm_to_head|elbow_to_head|shoulder_to_head|leg_to_body|unclear","actor":"offensive_player|defensive_player|both|unclear","target_body_area":"face|head|neck|torso|arm|leg|unclear","description":"string","confidence":"Low|Medium|High"}',
    "  ],",
    '  "possible_high_contact": false,',
    '  "high_contact_frame_ids": ["frame_00x"],',
    '  "high_contact_description": "string",',
    '  "high_contact_confidence": "Low|Medium|High",',
    '  "issue_cards": [',
    '    {"issue_id":"issue_001","issue_type":"blocking_charging|personal_foul|shooting_foul|high_contact|out_of_bounds|traveling|goaltending|unclear","description":"string","frame_ids":["frame_00x"],"confidence":"Low|Medium|High","needs_rule_retrieval":true,"rag_query":"string"}',
    "  ],",
    '  "officiating_issue_summary": "string",',
    '  "visual_frame_quality": "Good|Limited|Poor",',
    '  "verdict_readiness": "Ready|Limited|Not ready",',
    '  "can_reason_about_call": true,',
    '  "confidence_in_visual_description": "Low|Medium|High",',
    '  "missing_evidence": ["string"],',
    '  "limitations": ["string"],',
    '  "recommended_frames_for_verdict_agent": [{"frame_id":"frame_00x","timestamp":"M:SS.ss","reason":"string"}]',
    "}",
]

_CRITICAL_MOMENT_WHY_MAP = {
    "high motion": "possible contact",
    "pre-contact context": "defender position",
    "post-contact context": "possible contact",
    "clip start context": "other",
    "clip end context": "other",
    "temporal coverage": "other",
    "aftermath context": "possible contact",
    "player-on-floor context": "possible contact",
    "loose-ball context": "possible contact",
    "likely contact moment": "possible contact",
}


def _new_visual_result(model_name: str) -> dict:
    return {
        "agent": "visual_analyst",
        "success": False,
        "status": "Failed",
        "metadata": {
            "duration_seconds": None,
            "fps": None,
            "total_frames": None,
            "width": None,
            "height": None,
        },
        "evidence_frames": [],
        "evidence_selection_summary": "",
        "all_events_detected": [],
        "primary_event": {},
        "video_summary": "",
        "key_events": [],
        "possible_critical_moments": [],
        "sequence_interpretation": "unclear",
        "sequence_interpretation_reason": "",
        "primary_issue": "unclear",
        "primary_reason": "",
        "offense_team_color": "unknown",
        "defense_team_color": "unknown",
        "shooter_or_finisher": "unknown",
        "primary_contesting_defender": "unknown",
        "role_confidence": "low",
        "role_uncertainty_reason": "",
        "contact_direction_assessment": {},
        "secondary_issues": [],
        "visible_call_type": "unclear",
        "visible_call_type_reason": "",
        "possible_call_types": [],
        "contact_points": [],
        "possible_high_contact": False,
        "high_contact_frame_ids": [],
        "high_contact_description": "",
        "high_contact_confidence": "Low",
        "issue_cards": [],
        "officiating_issue_summary": "",
        "evidence_quality": "Poor",
        "visual_frame_quality": "Poor",
        "verdict_readiness": "Not Ready",
        "can_reason_about_call": False,
        "missing_evidence": [],
        "limitations": [],
        "recommended_frames_for_verdict_agent": [],
        "confidence_in_visual_description": "Low",
        "verdict": None,
        "original_call": "",
        "error": None,
        "error_type": None,
        "model_used": model_name,
        "debug": {
            "model_used": model_name,
            "primary_model_attempted": model_name,
            "fallback_model_attempted": _gemini_fallback_model(),
            "final_model_used": None,
            "retry_count": 0,
            "fallback_used": False,
            "primary_model_error_message": "",
            "real_video_frame_count": 0,
            "scan_fps_used": None,
            "dense_fps_used": None,
            "dense_window_seconds": None,
            "low_rate_candidates_count": 0,
            "dense_candidates_count": 0,
            "merged_candidates_count": 0,
            "motion_peaks_selected": [],
            "max_evidence_frames": _refcheck_max_evidence_frames(),
            "max_verdict_frames": _refcheck_max_verdict_frames(),
            "evidence_frame_selection_reasons": [],
            "chaotic_play_suspected": False,
            "candidates_scanned_count": 0,
            "boundary_candidates_count": 0,
            "dense_candidates_scanned_count": 0,
            "frames_extracted_count": 0,
            "frames_sent_to_gemini_count": 0,
            "selected_evidence_count": 0,
            "gemini_api_call_succeeded": False,
            "response_parsing_succeeded": False,
            "error_type": None,
            "error_message": None,
            "raw_response_excerpt": "",
            "scoring_top_candidates": [],
            "phase_bucket_counts": {},
            "phase_distribution_selected": {},
            "selection_mode": "",
        },
    }


def _set_failure(result: dict, error_type: str, message: str, debug_message: str | None = None) -> dict:
    result["error_type"] = error_type
    result["error"] = message
    result["debug"]["error_type"] = error_type
    result["debug"]["error_message"] = (debug_message if debug_message is not None else message)[:300]
    return result


def _open_video(video_path: str, result: dict):
    """Validate path and open a cv2 capture. Returns (cv2_module, capture) or (None, None) after mutating result."""
    if not video_path:
        _set_failure(result, "video_not_found", "Uploaded video path is missing.")
        return None, None

    video_file_path = Path(video_path)
    if not video_file_path.exists():
        _set_failure(result, "video_not_found", "Uploaded video file was not found.")
        return None, None

    try:
        import cv2
    except Exception as exc:
        _set_failure(result, "frame_extraction_failure", "Video processing dependency is unavailable.", str(exc))
        return None, None

    capture = cv2.VideoCapture(str(video_file_path))
    if not capture.isOpened():
        _set_failure(result, "frame_extraction_failure", "Uploaded video could not be opened.")
        return None, None

    return cv2, capture


def _read_video_metadata(capture, cv2_mod) -> tuple[float, int, dict]:
    fps = float(capture.get(cv2_mod.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2_mod.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2_mod.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2_mod.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    metadata = {
        "duration_seconds": round(duration, 2) if duration > 0 else None,
        "fps": round(fps, 2) if fps > 0 else None,
        "total_frames": total_frames if total_frames > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
    }
    return fps, total_frames, metadata


def _select_candidate_frames(capture, total_frames: int, fps: float, result: dict) -> tuple[list, str]:
    """Run scored selection with uniform fallback. Mutates result['debug']. Returns (candidates, mode)."""
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    scan_fps_used = _refcheck_scan_fps() if duration <= 20 else min(_refcheck_scan_fps(), 2.0)
    dense_fps_used = _refcheck_dense_fps()
    dense_window_seconds = _refcheck_dense_window_seconds()
    max_evidence_frames = min(_refcheck_max_evidence_frames(), _refcheck_max_gemini_images())
    min_evidence_frames = min(_refcheck_min_evidence_frames(), max_evidence_frames)
    result["debug"]["real_video_frame_count"] = total_frames
    result["debug"]["scan_fps_used"] = scan_fps_used
    result["debug"]["dense_fps_used"] = dense_fps_used
    result["debug"]["dense_window_seconds"] = dense_window_seconds
    result["debug"]["max_evidence_frames"] = max_evidence_frames
    result["debug"]["max_verdict_frames"] = _refcheck_max_verdict_frames()
    try:
        scan_indices = _candidate_indices(total_frames=total_frames, fps=fps, duration=duration)
        result["debug"]["candidates_scanned_count"] = len(scan_indices)
        result["debug"]["boundary_candidates_count"] = sum(
            1 for idx in scan_indices
            if (
                idx == 0
                or idx == total_frames - 1
                or idx <= int((total_frames - 1) * 0.08)
                or idx >= int((total_frames - 1) * 0.92)
            )
        )
        low_rate_candidates = _collect_candidates(capture=capture, indices=scan_indices, fps=fps)
        _tag_boundary_candidate_reasons(low_rate_candidates, total_frames)
        _score_candidates(low_rate_candidates)
        result["debug"]["low_rate_candidates_count"] = len(low_rate_candidates)
        if not low_rate_candidates:
            raise RuntimeError("No candidates after scan.")

        peaks = _rank_motion_peaks(low_rate_candidates, fps=fps, min_separation_seconds=1.0, max_peaks=5)
        try:
            dense_indices, peak_debug = _dense_indices_for_motion_peaks(
                peaks,
                total_frames=total_frames,
                fps=fps,
                dense_fps=dense_fps_used,
                window_seconds=dense_window_seconds,
            )
            dense_candidates = _collect_candidates(capture=capture, indices=dense_indices, fps=fps)
            _tag_dense_candidate_reasons(dense_candidates, peaks, fps=fps, window_seconds=dense_window_seconds)
            _score_candidates(dense_candidates)
        except Exception as exc:
            logger.warning("Dense frame extraction failed; using low-rate candidates only: %s", exc)
            dense_indices = []
            dense_candidates = []
            peak_debug = [
                {
                    "group_id": index,
                    "timestamp": peak.get("timestamp", ""),
                    "frame_idx": int(peak.get("frame_idx", 0)),
                    "motion": round(float(peak.get("motion", 0.0)), 5),
                    "motion_norm": round(float(peak.get("motion_norm", 0.0)), 3),
                    "score": round(float(peak.get("score", 0.0)), 3),
                }
                for index, peak in enumerate(peaks, start=1)
            ]

        candidates = _merge_candidates(low_rate_candidates, dense_candidates)
        _tag_boundary_candidate_reasons(candidates, total_frames)
        _score_candidates(candidates)
        candidates = _cap_candidate_pool(candidates, _refcheck_max_candidate_frames())
        _reindex_candidates(candidates)
        _tag_boundary_candidate_reasons(candidates, total_frames)
        _score_candidates(candidates)

        chaotic_play_suspected = _is_chaotic_play_suspected(candidates, peaks)
        if chaotic_play_suspected:
            max_evidence_frames = min(_refcheck_max_evidence_frames(), _refcheck_max_gemini_images())
        selected = _select_evidence_candidates(
            candidates,
            target_min=min_evidence_frames,
            target_max=max_evidence_frames,
            chaotic_play_suspected=chaotic_play_suspected,
        )
        if not selected:
            raise RuntimeError("Scoring produced no selected candidates.")

        result["debug"]["dense_candidates_scanned_count"] = len(dense_indices)
        result["debug"]["dense_candidates_count"] = len(dense_candidates)
        result["debug"]["merged_candidates_count"] = len(candidates)
        result["debug"]["motion_peaks_selected"] = peak_debug
        result["debug"]["chaotic_play_suspected"] = chaotic_play_suspected
        result["debug"]["evidence_frame_selection_reasons"] = [
            item.get("selection_reason", "selected for temporal coverage") for item in selected
        ]
        result["debug"]["phase_bucket_counts"] = _phase_bucket_counts(candidates)
        result["debug"]["phase_distribution_selected"] = _phase_bucket_counts(selected)
        selection_mode = "coverage_balanced_motion_selection"
        result["debug"]["selection_mode"] = selection_mode
        logger.info(
            "Frame selection: low_rate=%s dense=%s merged=%s peaks=%s selected=%s chaotic=%s",
            len(low_rate_candidates),
            len(dense_candidates),
            len(candidates),
            len(peaks),
            len(selected),
            chaotic_play_suspected,
        )
        result["debug"]["scoring_top_candidates"] = [
            {
                "timestamp": c["timestamp"],
                "reason": c.get("selection_reason", ""),
                "score": round(c.get("score", 0.0), 3),
            }
            for c in sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:5]
        ]
        return selected, selection_mode
    except Exception:
        fallback_indices = _uniform_fallback_indices(total_frames, desired=8)
        fallback_candidates = _collect_candidates(capture=capture, indices=fallback_indices, fps=fps)
        _tag_boundary_candidate_reasons(fallback_candidates, total_frames)
        for item in fallback_candidates:
            if item.get("selection_reason") not in {"clip start context", "clip end context"}:
                item["selection_reason"] = "temporal coverage"
            item["phase_bucket"] = _selection_phase(item.get("selection_reason", ""))
        result["debug"]["candidates_scanned_count"] = len(fallback_indices)
        result["debug"]["low_rate_candidates_count"] = len(fallback_candidates)
        result["debug"]["dense_candidates_count"] = 0
        result["debug"]["merged_candidates_count"] = len(fallback_candidates)
        result["debug"]["chaotic_play_suspected"] = False
        result["debug"]["phase_bucket_counts"] = _phase_bucket_counts(fallback_candidates)
        result["debug"]["phase_distribution_selected"] = _phase_bucket_counts(fallback_candidates)
        result["debug"]["selection_mode"] = "uniform_fallback"
        return fallback_candidates, "uniform_fallback"


def _persist_evidence_frames(selected_candidates: list, frames_dir: Path, cv2_mod) -> list[dict]:
    evidence_frames = []
    for i, candidate in enumerate(selected_candidates, start=1):
        frame_name = f"{uuid4().hex}_{i}.jpg"
        frame_path = frames_dir / frame_name
        wrote = cv2_mod.imwrite(str(frame_path), candidate["frame"], [int(cv2_mod.IMWRITE_JPEG_QUALITY), 92])
        if not wrote:
            continue
        evidence_frames.append(
            {
                "frame_id": f"frame_{i:03d}",
                "timestamp": candidate["timestamp"],
                "frame_url": f"{settings.MEDIA_URL}frames/{frame_name}",
                "frame_path": f"frames/{frame_name}",
                "local_path": str(frame_path),
                "selection_reason": candidate.get("selection_reason", "selected for temporal coverage"),
                "score": round(float(candidate.get("score", 0.0)), 4),
                "motion": round(float(candidate.get("motion", 0.0)), 5),
                "motion_norm": round(float(candidate.get("motion_norm", 0.0)), 4),
                "frame_context": candidate.get("frame_context", {}),
                "phase_bucket": candidate.get("phase_bucket") or _selection_phase(
                    candidate.get("selection_reason", "")
                ),
            }
        )
    return evidence_frames


def _build_selection_summary(
    selection_mode: str,
    scanned_count: int,
    num_evidence: int,
    debug: dict | None = None,
) -> str:
    debug = debug or {}
    if selection_mode == "coverage_balanced_motion_selection":
        peak_count = len(debug.get("motion_peaks_selected") or [])
        dense_fps = debug.get("dense_fps_used") or _refcheck_dense_fps()
        merged_count = debug.get("merged_candidates_count") or 0
        summary = (
            f"Scanned {scanned_count} whole-clip low-rate candidate frames, expanded dense windows around "
            f"{peak_count} motion peaks at {dense_fps:g} FPS, merged {merged_count} local candidates, "
            f"and selected {num_evidence} coverage-balanced evidence frames with start/end, before/contact/after, "
            "and temporal context."
        )
        if debug.get("chaotic_play_suspected"):
            summary += (
                " Chaotic contact pattern suspected, so the selector preserved additional aftermath "
                "and player-on-floor context frames."
            )
        return summary
    if selection_mode == "scored":
        return (
            f"Scanned {scanned_count} candidate frames and selected "
            f"{num_evidence} evidence frames using motion, sharpness, duplicate filtering, "
            "and temporal context around likely action moments."
        )
    return f"Scoring fallback triggered. Selected {num_evidence} uniformly distributed evidence frames."


def _initial_critical_moments(evidence_frames: list[dict]) -> list[dict]:
    moments = []
    for frame in evidence_frames[:6]:
        reason = frame.get("selection_reason", "selected for temporal coverage")
        moments.append(
            {
                "timestamp": frame["timestamp"],
                "frame_id": frame["frame_id"],
                "description": f"Frame selected as {reason}.",
                "why_relevant": _CRITICAL_MOMENT_WHY_MAP.get(reason, "other"),
            }
        )
    return moments


def _apply_missing_api_key(result: dict) -> None:
    result["error_type"] = "missing_api_key"
    result["error"] = "GEMINI_API_KEY is not configured. Frame extraction worked, but AI analysis is unavailable."
    result["limitations"] = ["AI description unavailable because Gemini API key is missing."]
    result["missing_evidence"] = [
        "No model-generated visual analyst interpretation was produced.",
        "Before/contact/after continuity cannot be assessed without model output.",
    ]
    result["officiating_issue_summary"] = "No neutral visual sequence summary is available."
    result["visible_call_type_reason"] = "Agent 1 does not classify the call without directly visible evidence."
    result["sequence_interpretation_reason"] = "No model output was available to distinguish a single continuous play from separate plays."


def _build_visual_prompt(evidence_frames: list[dict], original_call: str | None) -> str:
    lines = list(_VISUAL_PROMPT_INSTRUCTIONS)
    normalized_call = _normalize_original_call(original_call)
    if normalized_call:
        lines.append(f'User-provided original call: "{normalized_call}".')
        lines.append("Treat this as the primary review anchor unless direct visual evidence proves another issue is primary.")
        if _is_goaltending_call(normalized_call):
            lines.append(
                "Goaltending anchor: keep primary_issue as goaltending; evaluate ball trajectory, backboard timing, and cylinder position first. Any body contact should be reported as secondary unless overwhelmingly separate."
            )
    lines.append("Selected evidence frames in temporal order:")
    for frame in evidence_frames:
        lines.append(
            f'- {frame["frame_id"]} at {frame["timestamp"]} -> {frame.get("selection_reason", "context")}'
        )
    return "\n".join(lines)


def _call_gemini(api_key: str, model_name: str, prompt: str, evidence_frames: list[dict], result: dict) -> str:
    """Upload frames, call Gemini, mutate debug fields, return raw response text."""
    from google import genai

    client = genai.Client(api_key=api_key)
    evidence_frames = evidence_frames[: min(_refcheck_max_evidence_frames(), _refcheck_max_gemini_images())]
    uploaded_parts = [client.files.upload(file=frame["local_path"]) for frame in evidence_frames]
    result["debug"]["frames_sent_to_gemini_count"] = len(uploaded_parts)
    raw_text = _generate_gemini_with_fallback(
        client=client,
        primary_model=model_name,
        fallback_model=_gemini_fallback_model(),
        contents=[prompt, *uploaded_parts],
        debug=result["debug"],
    )
    result["debug"]["gemini_api_call_succeeded"] = True
    final_model = result["debug"].get("final_model_used") or model_name
    result["model_used"] = final_model
    result["debug"]["model_used"] = final_model
    result["debug"]["raw_response_excerpt"] = _safe_excerpt(raw_text)
    return raw_text


def _apply_empty_response(result: dict) -> None:
    result["error_type"] = "empty_gemini_response"
    result["error"] = "AI returned an empty response."
    result["limitations"] = ["The AI response was empty for this request."]
    result["missing_evidence"] = [
        "No visual analyst sequence interpretation was returned.",
        "Unable to assess whether evidence supports later verdict reasoning.",
    ]
    result["officiating_issue_summary"] = "No neutral visual sequence summary was returned."
    result["visible_call_type_reason"] = "No model output available."
    result["sequence_interpretation_reason"] = "No model output was available to assess sequence continuity."


def _apply_unstructured_response(result: dict, raw_text: str) -> None:
    result["error_type"] = "json_parsing_failure"
    result["video_summary"] = _strip_code_fences(raw_text)[:900]
    result["limitations"] = [
        "Structured JSON parsing failed; showing raw AI summary text instead.",
        "Selected frames may be insufficient for complete temporal understanding.",
    ]
    result["missing_evidence"] = [
        "Structured visual analyst fields were not returned in JSON.",
        "Frame-linked critical moments and recommendations are uncertain.",
    ]
    result["officiating_issue_summary"] = "Unstructured output; visual sequence assessment is uncertain."
    result["visible_call_type_reason"] = "Unstructured model output did not provide a reliable visual basis."
    result["sequence_interpretation_reason"] = "Unstructured output did not provide a reliable sequence interpretation."
    result["error"] = "AI returned unstructured text."


def _apply_json_decode_failure(result: dict, raw_text: str, exc: Exception) -> None:
    result["error_type"] = "json_parsing_failure"
    result["video_summary"] = _strip_code_fences(raw_text)[:900]
    result["limitations"] = [
        "Structured JSON parsing failed; showing raw AI summary text instead.",
        "Some expected visual analyst fields may be missing.",
    ]
    result["missing_evidence"] = [
        "Frame-linked event fields could not be parsed.",
        "Evidence sufficiency judgement remains uncertain.",
    ]
    result["officiating_issue_summary"] = "JSON parsing failed; visual description is incomplete."
    result["visible_call_type_reason"] = "Parsing failure prevented reliable visual-basis extraction."
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = str(exc)[:300]


def _apply_gemini_api_failure(result: dict, exc: Exception) -> None:
    result["error_type"] = "gemini_api_failure"
    result["error"] = "AI analysis could not be completed for this upload."
    result["limitations"] = ["Gemini API request failed before structured analysis could be produced."]
    result["missing_evidence"] = [
        "No complete visual analyst output was received.",
        "Cannot determine if evidence is sufficient for a later verdict agent.",
    ]
    result["officiating_issue_summary"] = "Gemini request failed; visual analyst stage not completed."
    result["visible_call_type_reason"] = "No visual basis available due to API failure."
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = str(exc)[:300]


def _apply_empty_summary(result: dict) -> None:
    result["error_type"] = "empty_gemini_response"
    result["error"] = "AI response was received but summary text was empty."
    result["limitations"] = ["AI returned structured data without a usable summary."]
    result["missing_evidence"] = [
        "No usable visual analyst summary was returned.",
        "Unable to confirm continuity around potential call moment.",
    ]
    result["officiating_issue_summary"] = "Insufficient visual summary for later verdict reasoning."
    result["visible_call_type_reason"] = "No reliable visual basis returned."


def _normalize_key_events(payload_events: list, frame_id_map: dict, evidence_frames: list[dict]) -> list[dict]:
    normalized = []
    for event in payload_events[:8]:
        if not isinstance(event, dict):
            continue
        timestamp = str(event.get("timestamp", "")).strip()
        frame_id = str(event.get("frame_id", "")).strip()
        description = _complete_sentence(str(event.get("description", "")).strip())
        if not description:
            continue
        if frame_id not in frame_id_map:
            frame_id = evidence_frames[0]["frame_id"] if evidence_frames else "frame_001"
        if not timestamp:
            timestamp = frame_id_map.get(frame_id, {}).get("timestamp", "")
        normalized.append({"timestamp": timestamp[:20], "frame_id": frame_id, "description": description})
    return normalized


def _normalize_critical_moments(payload_critical: list, frame_id_map: dict, evidence_frames: list[dict]) -> list[dict]:
    normalized = []
    for item in payload_critical[:10]:
        if not isinstance(item, dict):
            continue
        timestamp = str(item.get("timestamp", "")).strip()
        frame_id = str(item.get("frame_id", "")).strip()
        description = _complete_sentence(str(item.get("description", "")).strip())
        why = _normalize_why_relevant(item.get("why_relevant", "other"))
        if not description:
            continue
        if frame_id not in frame_id_map:
            frame_id = evidence_frames[0]["frame_id"] if evidence_frames else "frame_001"
        if not timestamp:
            timestamp = frame_id_map.get(frame_id, {}).get("timestamp", "")
        normalized.append(
            {
                "timestamp": timestamp[:20],
                "frame_id": frame_id,
                "description": description,
                "why_relevant": why,
            }
        )
    return normalized


def _normalize_recommended_frames(payload_recommended: list, frame_id_map: dict) -> list[dict]:
    normalized = []
    for item in payload_recommended[:8]:
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("frame_id", "")).strip()
        timestamp = str(item.get("timestamp", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if frame_id not in frame_id_map:
            continue
        if not timestamp:
            timestamp = frame_id_map[frame_id]["timestamp"]
        if not reason:
            reason = "Useful visual evidence for later verdict reasoning."
        normalized.append({"frame_id": frame_id, "timestamp": timestamp[:20], "reason": reason[:220]})
    return normalized


def _normalize_possible_call_types(payload_call_types: list) -> list[str]:
    normalized: list[str] = []
    for call_type in payload_call_types[:6]:
        candidate = _normalize_possible_call_type(str(call_type))
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _normalize_contact_points(payload_contacts: list, frame_id_map: dict, evidence_frames: list[dict]) -> list[dict]:
    if not isinstance(payload_contacts, list):
        return []
    normalized = []
    for item in payload_contacts[:12]:
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("frame_id", "")).strip()
        if frame_id not in frame_id_map:
            frame_id = evidence_frames[0]["frame_id"] if evidence_frames else "frame_001"
        timestamp = str(item.get("timestamp", "")).strip() or frame_id_map.get(frame_id, {}).get("timestamp", "")
        description = _complete_sentence(str(item.get("description", "")).strip(), max_len=350)
        if not description:
            continue
        normalized.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp[:20],
                "contact_type": _normalize_contact_type(item.get("contact_type", "")),
                "actor": _normalize_contact_actor(item.get("actor", "")),
                "target_body_area": _normalize_body_area(item.get("target_body_area", "")),
                "description": description,
                "confidence": _normalize_confidence(item.get("confidence", "")),
            }
        )
    return normalized


def _normalize_issue_cards(payload_issues: list, frame_id_map: dict) -> list[dict]:
    if not isinstance(payload_issues, list):
        return []
    normalized = []
    seen_ids = set()
    for index, item in enumerate(payload_issues[:10], start=1):
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id", "")).strip() or f"issue_{index:03d}"
        if issue_id in seen_ids:
            issue_id = f"{issue_id}_{index}"
        frame_ids = []
        for frame_id in item.get("frame_ids") or []:
            candidate = str(frame_id).strip()
            if candidate in frame_id_map and candidate not in frame_ids:
                frame_ids.append(candidate)
        description = _complete_sentence(str(item.get("description", "")).strip(), max_len=450)
        rag_query = " ".join(str(item.get("rag_query", "")).strip().split())[:300]
        if not description and not rag_query:
            continue
        normalized.append(
            {
                "issue_id": issue_id,
                "issue_type": _normalize_issue_type(item.get("issue_type", "")),
                "description": description or "Plausible officiating issue identified from selected frames.",
                "frame_ids": frame_ids,
                "confidence": _normalize_confidence(item.get("confidence", "")),
                "needs_rule_retrieval": bool(item.get("needs_rule_retrieval", True)),
                "rag_query": rag_query,
            }
        )
        seen_ids.add(issue_id)
    return normalized


def _normalize_event_frame_ids(raw_frame_ids, frame_id_map: dict) -> list[str]:
    frame_ids = []
    if not isinstance(raw_frame_ids, list):
        return frame_ids
    for frame_id in raw_frame_ids[:8]:
        candidate = str(frame_id).strip()
        if candidate in frame_id_map and candidate not in frame_ids:
            frame_ids.append(candidate)
    return frame_ids


def _normalize_detected_events(payload_events: list, frame_id_map: dict) -> list[dict]:
    if not isinstance(payload_events, list):
        return []
    normalized = []
    for item in payload_events[:12]:
        if not isinstance(item, dict):
            continue
        description = _complete_sentence(str(item.get("brief_description", "")).strip(), max_len=350)
        frame_ids = _normalize_event_frame_ids(item.get("frame_ids") or [], frame_id_map)
        timestamp = str(item.get("timestamp", "")).strip()
        if not description:
            continue
        normalized.append(
            {
                "tier": _normalize_event_tier(item.get("tier", "")),
                "timestamp": timestamp[:20],
                "frame_ids": frame_ids,
                "brief_description": description,
            }
        )
    normalized.sort(
        key=lambda event: (
            _timestamp_to_seconds(event.get("timestamp", "")),
            -_event_tier_rank(event.get("tier", "")),
        )
    )
    return normalized


def _normalize_primary_event(payload_event: dict, frame_id_map: dict) -> dict:
    if not isinstance(payload_event, dict):
        return {}
    description = _complete_sentence(str(payload_event.get("description", "")).strip(), max_len=500)
    why = _complete_sentence(str(payload_event.get("why_this_is_primary", "")).strip(), max_len=500)
    if not description and not why:
        return {}
    return {
        "tier": _normalize_event_tier(payload_event.get("tier", "")),
        "timestamp": str(payload_event.get("timestamp", "")).strip()[:20],
        "frame_ids": _normalize_event_frame_ids(payload_event.get("frame_ids") or [], frame_id_map),
        "description": description or "Primary visible officiating event.",
        "why_this_is_primary": why,
    }


def _primary_event_from_detected_events(events: list[dict]) -> dict:
    if not events:
        return {}
    selected = max(
        events,
        key=lambda event: (
            _event_tier_rank(event.get("tier", "")),
            _timestamp_to_seconds(event.get("timestamp", "")),
        ),
    )
    return {
        "tier": selected["tier"],
        "timestamp": selected.get("timestamp", ""),
        "frame_ids": selected.get("frame_ids", []),
        "description": selected.get("brief_description", ""),
        "why_this_is_primary": _complete_sentence(
            "Selected because it is the highest-tier detected event; same-tier ties are resolved toward the latest event that most directly defines the action."
        ),
    }


def _default_issue_cards(result: dict, evidence_frames: list[dict]) -> list[dict]:
    issue_cards = []
    if result.get("possible_call_types"):
        for index, call_type in enumerate(result["possible_call_types"][:4], start=1):
            issue_type = "blocking_charging" if call_type == "blocking/charging" else call_type.replace(" ", "_")
            issue_cards.append(
                {
                    "issue_id": f"issue_{index:03d}",
                    "issue_type": _normalize_issue_type(issue_type),
                    "description": _complete_sentence(f"Visual evidence may involve {call_type} considerations."),
                    "frame_ids": [frame["frame_id"] for frame in evidence_frames[:4]],
                    "confidence": result.get("confidence_in_visual_description", "Low"),
                    "needs_rule_retrieval": True,
                    "rag_query": f"basketball {call_type} rule visual contact positioning",
                }
            )
    if result.get("possible_high_contact"):
        issue_cards.append(
            {
                "issue_id": f"issue_{len(issue_cards) + 1:03d}",
                "issue_type": "high_contact",
                "description": result.get("high_contact_description") or "Possible contact near the face, head, or neck area is visible or cannot be ruled out.",
                "frame_ids": result.get("high_contact_frame_ids") or [frame["frame_id"] for frame in evidence_frames[:4]],
                "confidence": result.get("high_contact_confidence", "Low"),
                "needs_rule_retrieval": True,
                "rag_query": "basketball contact to head face neck illegal hand arm contact personal foul flagrant criteria",
            }
        )
    return issue_cards


def _apply_payload_to_result(result: dict, payload: dict, evidence_frames: list[dict]) -> None:
    frame_id_map = {frame["frame_id"]: frame for frame in evidence_frames}

    normalized_events = _normalize_key_events(payload.get("key_events") or [], frame_id_map, evidence_frames)
    normalized_critical = _normalize_critical_moments(
        payload.get("possible_critical_moments") or [], frame_id_map, evidence_frames
    )
    normalized_recommended = _normalize_recommended_frames(
        payload.get("recommended_frames_for_verdict_agent") or [], frame_id_map
    )
    normalized_limitations = [
        str(item).strip()[:220] for item in (payload.get("limitations") or [])[:6] if str(item).strip()
    ]
    normalized_missing = [
        str(item).strip()[:220] for item in (payload.get("missing_evidence") or [])[:6] if str(item).strip()
    ]
    normalized_contacts = _normalize_contact_points(payload.get("contact_points") or [], frame_id_map, evidence_frames)
    normalized_issue_cards = _normalize_issue_cards(payload.get("issue_cards") or [], frame_id_map)
    normalized_detected_events = _normalize_detected_events(payload.get("all_events_detected") or [], frame_id_map)
    normalized_primary_event = _normalize_primary_event(payload.get("primary_event") or {}, frame_id_map)
    normalized_secondary_issues = _normalize_secondary_issues(payload.get("secondary_issues") or [])
    normalized_contact_direction = _normalize_contact_direction(payload.get("contact_direction_assessment") or {})
    normalized_original_call = _normalize_original_call(result.get("original_call") or payload.get("original_call"))
    goaltending_selected = _is_goaltending_call(normalized_original_call)
    if not normalized_primary_event:
        normalized_primary_event = _primary_event_from_detected_events(normalized_detected_events)

    result["agent"] = str(payload.get("agent", "visual_analyst")).strip().lower() or "visual_analyst"
    result["original_call"] = normalized_original_call
    result["all_events_detected"] = normalized_detected_events
    result["primary_event"] = normalized_primary_event
    result["video_summary"] = str(payload.get("video_summary", "")).strip()[:900]
    result["key_events"] = normalized_events
    result["possible_critical_moments"] = normalized_critical or result["possible_critical_moments"]

    sequence_interpretation = _normalize_sequence_interpretation(payload.get("sequence_interpretation", ""))
    sequence_reason = str(
        payload.get("sequence_interpretation_reason", "")
    ).strip()[:500]
    if sequence_interpretation == "unclear":
        sequence_interpretation = "single_play"
    if not sequence_reason:
        if sequence_interpretation == "single_play":
            sequence_reason = "Selected frames are treated as one continuous play unless they clearly show separate plays."
        elif sequence_interpretation == "uncertain_continuity":
            sequence_reason = "Continuity is uncertain in isolated frames."
    result["sequence_interpretation"] = sequence_interpretation
    result["sequence_interpretation_reason"] = sequence_reason
    result["visible_call_type"] = _normalize_call_type(payload.get("visible_call_type", ""))
    result["visible_call_type_reason"] = str(payload.get("visible_call_type_reason", "")).strip()[:500]
    result["possible_call_types"] = _normalize_possible_call_types(payload.get("possible_call_types") or [])
    if goaltending_selected and "goaltending" not in result["possible_call_types"]:
        result["possible_call_types"] = ["goaltending", *result["possible_call_types"]][:6]
    result["contact_points"] = normalized_contacts
    primary_issue = _normalize_call_type(payload.get("primary_issue", ""))
    if primary_issue == "unclear":
        if goaltending_selected:
            primary_issue = "goaltending"
        elif result["visible_call_type"] != "unclear":
            primary_issue = result["visible_call_type"]
        elif result["possible_call_types"]:
            primary_issue = result["possible_call_types"][0]
    result["primary_issue"] = primary_issue
    result["primary_reason"] = _complete_sentence(str(payload.get("primary_reason", "")).strip(), max_len=500)
    result["offense_team_color"] = str(payload.get("offense_team_color", "")).strip()[:60] or "unknown"
    result["defense_team_color"] = str(payload.get("defense_team_color", "")).strip()[:60] or "unknown"
    result["shooter_or_finisher"] = str(payload.get("shooter_or_finisher", "")).strip()[:140] or "unknown"
    result["primary_contesting_defender"] = (
        str(payload.get("primary_contesting_defender", "")).strip()[:140] or "unknown"
    )
    result["role_confidence"] = _normalize_role_confidence(payload.get("role_confidence", "low"), default="low")
    result["role_uncertainty_reason"] = str(payload.get("role_uncertainty_reason", "")).strip()[:320]
    result["contact_direction_assessment"] = normalized_contact_direction
    result["secondary_issues"] = normalized_secondary_issues
    result["possible_high_contact"] = bool(payload.get("possible_high_contact", False))
    result["high_contact_frame_ids"] = [
        str(frame_id).strip()
        for frame_id in (payload.get("high_contact_frame_ids") or [])[:8]
        if str(frame_id).strip() in frame_id_map
    ]
    result["high_contact_description"] = _complete_sentence(
        str(payload.get("high_contact_description", "")).strip(), max_len=500
    )
    result["high_contact_confidence"] = _normalize_confidence(payload.get("high_contact_confidence", ""))
    if not result["possible_high_contact"]:
        high_contact_contacts = [
            contact for contact in normalized_contacts
            if contact["target_body_area"] in {"face", "head", "neck"}
            or contact["contact_type"] in {"hand_to_face", "arm_to_head", "elbow_to_head", "shoulder_to_head"}
        ]
        result["possible_high_contact"] = bool(high_contact_contacts)
        if high_contact_contacts:
            result["high_contact_frame_ids"] = sorted({contact["frame_id"] for contact in high_contact_contacts})
            result["high_contact_description"] = high_contact_contacts[0]["description"]
            result["high_contact_confidence"] = high_contact_contacts[0]["confidence"]

    if goaltending_selected:
        result["primary_issue"] = "goaltending"
        if not result["primary_reason"]:
            result["primary_reason"] = _complete_sentence(
                "Goaltending remains the primary issue because it is the user-selected initial call and the review depends on ball trajectory, backboard timing, and cylinder position evidence."
            )
        if result["visible_call_type"] == "unclear":
            result["visible_call_type"] = "goaltending"
            if not result["visible_call_type_reason"]:
                result["visible_call_type_reason"] = (
                    "Using user-provided original call as review anchor; Agent 1 is not issuing a rules conclusion."
                )

        goaltending_issue_cards = []
        demoted_issue_cards = []
        for issue in normalized_issue_cards:
            if issue.get("issue_type") == "goaltending":
                goaltending_issue_cards.append(issue)
            else:
                demoted_issue_cards.append(issue)

        if not goaltending_issue_cards:
            goaltending_frame_ids = normalized_primary_event.get("frame_ids") or [
                frame.get("frame_id", "")
                for frame in (normalized_recommended or _default_recommended_frames(evidence_frames))[:4]
                if frame.get("frame_id")
            ]
            if not goaltending_frame_ids:
                goaltending_frame_ids = [frame.get("frame_id", "") for frame in evidence_frames[:4] if frame.get("frame_id")]
            goaltending_issue_cards = [
                {
                    "issue_id": "issue_001",
                    "issue_type": "goaltending",
                    "description": _complete_sentence(
                        "Determine whether defender-ball touch occurred on downward flight, after backboard contact, or above/inside the cylinder."
                    ),
                    "frame_ids": goaltending_frame_ids[:6],
                    "confidence": result.get("confidence_in_visual_description", "Low"),
                    "needs_rule_retrieval": True,
                    "rag_query": "NBA goaltending basket interference downward flight backboard contact cylinder",
                }
            ]

        for issue in demoted_issue_cards:
            result["secondary_issues"].append(
                {
                    "issue": _complete_sentence(issue.get("description", ""), max_len=220),
                    "confidence": _normalize_role_confidence(issue.get("confidence", "low_to_medium"), default="low_to_medium"),
                    "reason": _complete_sentence(
                        "Visible contact or positioning may matter for a separate foul analysis, but it is secondary to the selected goaltending review.",
                        max_len=300,
                    ),
                }
            )
        result["issue_cards"] = goaltending_issue_cards[:4]
    else:
        result["issue_cards"] = normalized_issue_cards

    result["officiating_issue_summary"] = str(payload.get("officiating_issue_summary", "")).strip()[:500]
    result["visual_frame_quality"] = _normalize_evidence_quality(
        payload.get("visual_frame_quality", payload.get("evidence_quality", ""))
    )
    result["evidence_quality"] = result["visual_frame_quality"]
    result["verdict_readiness"] = _normalize_verdict_readiness(payload.get("verdict_readiness", ""))

    requested_can_reason = bool(payload.get("can_reason_about_call", False))
    result["can_reason_about_call"] = (
        requested_can_reason
        and result["verdict_readiness"] == "Ready"
        and result["sequence_interpretation"] != "uncertain_continuity"
    )
    result["missing_evidence"] = normalized_missing
    if goaltending_selected and (not result["can_reason_about_call"] or result["verdict_readiness"] != "Ready"):
        goaltending_missing_hints = [
            "Exact frame of defender-ball contact.",
            "Clear ball trajectory before and after contact.",
            "Whether ball touched backboard before defender contact.",
            "Whether ball was on downward flight.",
            "Whether ball was above or inside the cylinder.",
        ]
        for hint in goaltending_missing_hints:
            if hint not in result["missing_evidence"]:
                result["missing_evidence"].append(hint)
        result["missing_evidence"] = result["missing_evidence"][:8]
    if requested_can_reason and not result["can_reason_about_call"]:
        if result["verdict_readiness"] != "Ready":
            result["missing_evidence"].append(
                "Verdict readiness is not marked Ready because the selected frames do not provide enough continuous visual context."
            )
    result["limitations"] = normalized_limitations
    result["recommended_frames_for_verdict_agent"] = (
        normalized_recommended if normalized_recommended else _default_recommended_frames(evidence_frames)
    )
    result["confidence_in_visual_description"] = _normalize_confidence(
        payload.get("confidence_in_visual_description", "")
    )
    payload_selection_summary = str(payload.get("evidence_selection_summary", "")).strip()
    if payload_selection_summary:
        result["evidence_selection_summary"] = payload_selection_summary[:400]
    if not result["issue_cards"]:
        result["issue_cards"] = _default_issue_cards(result, evidence_frames)
    if goaltending_selected and result["issue_cards"]:
        result["issue_cards"] = [issue for issue in result["issue_cards"] if issue.get("issue_type") == "goaltending"] or result["issue_cards"][:1]


def _run_gemini_analysis(
    api_key: str,
    model_name: str,
    evidence_frames: list[dict],
    original_call: str | None,
    result: dict,
) -> None:
    """Run the full Gemini call + parse pipeline, mutating result in place."""
    prompt = _build_visual_prompt(evidence_frames, original_call)
    raw_text = ""
    try:
        raw_text = _call_gemini(api_key, model_name, prompt, evidence_frames, result)

        if not raw_text:
            _apply_empty_response(result)
            return

        json_candidate = _extract_json_object(raw_text)
        if not json_candidate:
            _apply_unstructured_response(result, raw_text)
            return

        payload = json.loads(json_candidate)
        result["debug"]["response_parsing_succeeded"] = True
        _apply_payload_to_result(result, payload, evidence_frames)

        if not result["video_summary"]:
            _apply_empty_summary(result)
            return

        result["success"] = True
        result["status"] = "Analyzed"
        result["error"] = None
        result["error_type"] = None
    except json.JSONDecodeError as exc:
        _apply_json_decode_failure(result, raw_text, exc)
    except Exception as exc:
        _apply_gemini_api_failure(result, exc)


# ----------------------------------------------------------------------------
# Rulebook retrieval (OpenAI embedding -> Pinecone "nba-rules")
# ----------------------------------------------------------------------------

def _embed_summary_openai(summary: str) -> list[float] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model="text-embedding-3-small", input=summary)
        return list(response.data[0].embedding)
    except Exception:
        return None


def _query_pinecone_rules(vector: list[float], top_k: int = 5) -> list[dict]:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return []
    try:
        from pinecone import Pinecone

        index_name = os.getenv("PINECONE_INDEX_NAME", "nba-rules")
        namespace = os.getenv("PINECONE_NAMESPACE", "") or None
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        query_kwargs = {"vector": vector, "top_k": top_k, "include_metadata": True}
        if namespace:
            query_kwargs["namespace"] = namespace
        response = index.query(**query_kwargs)
    except Exception:
        return []

    raw_matches = []
    if isinstance(response, dict):
        raw_matches = response.get("matches") or []
    else:
        raw_matches = getattr(response, "matches", None) or []

    normalized: list[dict] = []
    for match in raw_matches:
        if isinstance(match, dict):
            match_id = match.get("id", "")
            score = match.get("score", 0.0)
            metadata = match.get("metadata") or {}
        else:
            match_id = getattr(match, "id", "")
            score = getattr(match, "score", 0.0)
            metadata = getattr(match, "metadata", None) or {}
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            score_val = 0.0
        text = str(metadata.get("text") or metadata.get("chunk") or metadata.get("content") or "").strip()
        section = str(metadata.get("section") or metadata.get("rule") or metadata.get("rule_number") or "").strip()
        entry = {
            "rule_id": str(match_id),
            "score": score_val,
            "text": text[:600],
            "section": section[:120],
        }
        if not text:
            entry["meta"] = {k: str(v)[:200] for k, v in metadata.items()}
        normalized.append(entry)
    return normalized


def _retrieve_matching_rules(video_summary: str, top_k: int = 5) -> list[dict]:
    summary = (video_summary or "").strip()
    if not summary:
        return []
    try:
        vector = _embed_summary_openai(summary)
        if not vector:
            return []
        return _query_pinecone_rules(vector, top_k=top_k)
    except Exception:
        return []


def _rule_query_for_contact(contact: dict) -> str:
    parts = [
        "basketball officiating",
        contact.get("contact_type", ""),
        contact.get("target_body_area", ""),
        "contact rule personal foul illegal contact",
    ]
    if contact.get("target_body_area") in {"face", "head", "neck"}:
        parts.append("head face neck contact flagrant criteria")
    return " ".join(part for part in parts if part).strip()


def _dedupe_rules(rules: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for rule in sorted(rules, key=lambda item: item.get("score", 0.0), reverse=True):
        key = rule.get("rule_id") or (rule.get("section"), rule.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def retrieve_rules_for_issues(visual_result: dict) -> dict:
    grouped: dict[str, list[dict]] = {}

    queries_by_issue: dict[str, list[str]] = {}
    video_summary = (visual_result.get("video_summary") or "").strip()
    if video_summary:
        queries_by_issue.setdefault("summary", []).append(video_summary)

    possible_call_types = visual_result.get("possible_call_types") or []
    for call_type in possible_call_types:
        queries_by_issue.setdefault(f"call_type_{call_type.replace('/', '_').replace(' ', '_')}", []).append(
            f"basketball {call_type} officiating rule"
        )

    original_call = _normalize_original_call(visual_result.get("original_call"))
    if _is_goaltending_call(original_call) or visual_result.get("primary_issue") == "goaltending":
        goaltending_queries = [
            "NBA goaltending rule downward flight",
            "NBA basket interference cylinder rule",
            "NBA defender touches ball after backboard contact rule",
            "NBA goaltending exact point of ball contact evidence",
        ]
        for query in goaltending_queries:
            queries_by_issue.setdefault("selected_call_goaltending", []).append(query)

    for issue in visual_result.get("issue_cards") or []:
        if not isinstance(issue, dict) or not issue.get("needs_rule_retrieval", True):
            continue
        issue_id = str(issue.get("issue_id") or "issue").strip()
        query = str(issue.get("rag_query") or issue.get("description") or "").strip()
        if query:
            queries_by_issue.setdefault(issue_id, []).append(query)

    for index, contact in enumerate(visual_result.get("contact_points") or [], start=1):
        if not isinstance(contact, dict):
            continue
        issue_id = "high_contact" if contact.get("target_body_area") in {"face", "head", "neck"} else f"contact_{index:03d}"
        queries_by_issue.setdefault(issue_id, []).append(_rule_query_for_contact(contact))

    if visual_result.get("possible_high_contact"):
        high_contact_queries = [
            "basketball personal foul contact to head face neck",
            "basketball illegal hand arm contact to face head neck",
            "basketball shooting foul contact to head face neck",
            "basketball flagrant foul criteria head face neck contact",
            visual_result.get("high_contact_description", ""),
        ]
        for query in high_contact_queries:
            if query:
                queries_by_issue.setdefault("high_contact", []).append(query)

    for issue_id, queries in queries_by_issue.items():
        rules = []
        for query in queries[:6]:
            rules.extend(_retrieve_matching_rules(query, top_k=3))
        grouped[issue_id] = _dedupe_rules(rules)[:6]
    return grouped


def _cap_grouped_rules_total(grouped_rules: dict, max_rules: int = 3) -> dict:
    if max_rules <= 0:
        return {}
    flattened = []
    for issue_id, rules in (grouped_rules or {}).items():
        for rule in rules or []:
            flattened.append((issue_id, rule))
    capped = _dedupe_rules([rule for _, rule in flattened])[:max_rules]
    capped_keys = {
        rule.get("rule_id") or (rule.get("section"), rule.get("text"))
        for rule in capped
    }
    capped_by_issue = {}
    for issue_id, rule in flattened:
        key = rule.get("rule_id") or (rule.get("section"), rule.get("text"))
        if key not in capped_keys:
            continue
        capped_by_issue.setdefault(issue_id, []).append(rule)
        capped_keys.remove(key)
        if not capped_keys:
            break
    return capped_by_issue


# ----------------------------------------------------------------------------
# Agent 2: Verdict Agent
# ----------------------------------------------------------------------------

_VERDICT_PROMPT_INSTRUCTIONS = [
    "# ROLE",
    "You are Agent 2: Verdict Agent in a multi-agent officiating-review pipeline.",
    "You receive (a) a neutral visual description from Agent 1 (the Visual Analyst) and (b) a small set of curated still frames Agent 1 recommended for verdict review.",
    "Your job is to reason issue-by-issue, then output whether the original call appears Fair, Bad, or Inconclusive when an original call is known.",
    "If the original call is unknown or not provided, do not output Fair Call or Bad Call as a comparison. Output Inconclusive and include correct_ruling_if_any if the visuals support a suggested ruling.",
    "",
    "# YOUR JOB",
    "Evaluate the selected original_call first and keep verdict reasoning anchored to that call type.",
    "When original_call is Goaltending, verdict must depend on ball trajectory, backboard timing, cylinder position, and defender-ball touch visibility. Body contact may be listed as other_possible_issues but must not replace the selected-call analysis.",
    "Before blocking/charging reasoning, inspect issue_cards, possible_high_contact, and contact_points. Treat high contact as a separate issue from body-position or charge/block analysis.",
    "Integrate the visual evidence with standard NBA officiating expectations, including but not limited to: blocking vs. charging, traveling, goaltending and basket interference, shooting fouls, personal fouls, out-of-bounds, three-second / defensive three-second, and restricted-area rules.",
    "Cite frames by frame_id when stating what you see. You may quote short phrases from the Agent 1 video_summary when they support a claim.",
    "",
    "# VERDICT LABELS",
    "Output exactly one of: 'Fair Call', 'Bad Call', 'Inconclusive'.",
    "- Fair Call: the visible evidence supports the call that would normally be made on this play.",
    "- Bad Call: the visible evidence clearly contradicts the call that would normally be made on this play.",
    "- Inconclusive: visible evidence does not clearly support either side, OR key continuity is missing (e.g., defender feet pre-contact not visible, exact moment of contact not captured, ball release timing unclear, foot-on-line not visible).",
    "When in doubt, choose Inconclusive. Do not guess.",
    "",
    "# RULES OF REASONING",
    "- Never invent details that are not present in video_summary or the provided frames.",
    "- Never assume what happened between frames. The frames are discrete stills, not continuous footage.",
    "- Be specific about what the frames literally show vs. what is inferred.",
    "- If a key element (e.g., a defender's feet) is not visible in any provided frame, say so under limitations and lean toward Inconclusive.",
    "- Use original_call only as the call being evaluated. Do not treat it as visual evidence.",
    "- Separate two things clearly: (1) selected_call_reasoning for the original_call, and (2) other_possible_issues that are visible but secondary.",
    "- For goaltending reviews, if ball/rim/backboard/cylinder evidence is missing or ambiguous, verdict should be Inconclusive even when body contact is visible.",
    "- If possible_high_contact is true, do not conclude Fair Call solely because a defender appears set.",
    "- If high contact is visible or possible but not clear enough, prefer Inconclusive or Medium/Low confidence.",
    "- If Agent 1 confidence_in_visual_description is Low, your confidence cannot be High.",
    "- If key contact frames have motion blur, partial occlusion, or unclear face/head/neck visibility, your confidence cannot be High.",
    "- If original_call is missing or unknown, verdict must be Inconclusive.",
    "- Confidence reflects how strongly the visible evidence supports your verdict label, not how certain you feel in general. Use Low when evidence is thin, Medium when partial, High only when the frames clearly settle the question.",
    "",
    "# OUTPUT REQUIREMENTS",
    "- All descriptions must be complete sentences.",
    "- key_factors must cite the frame_id (and timestamp) the factor is grounded in.",
    "- reasoning must be a single short paragraph of 3 to 6 sentences.",
    "- key_factors must contain between 2 and 5 entries.",
    "- issue_analysis must contain one entry for each important issue_card, including high_contact when present.",
    "- rule_basis must explicitly mention the retrieved database rule section or rule_id that drove the verdict when rulebook context is provided.",
    "- relevant_rules_used must list the most relevant retrieved database rules you relied on. Use only rule_id/section values present in the provided Rulebook context.",
    "- limitations must list anything that constrained the verdict (occluded body parts, missing pre/post-contact frames, ambiguous body part of contact, etc.).",
    "Return valid JSON only, with no prose outside the JSON, no markdown fences, and exactly this schema:",
    "{",
    '  "agent": "verdict_agent",',
    '  "original_call": "string",',
    '  "verdict": "Fair Call|Bad Call|Inconclusive",',
    '  "confidence": "Low|Medium|High",',
    '  "correct_ruling_if_any": "string",',
    '  "reasoning": "string",',
    '  "selected_call_reasoning": "string",',
    '  "other_possible_issues": [{"issue":"string","confidence":"low|medium|high|low_to_medium|medium_to_high","reason":"string"}],',
    '  "issue_analysis": [{"issue_id":"issue_001","issue_type":"string","finding":"string","confidence":"Low|Medium|High","relevant_frames":["frame_00x"],"relevant_rules":["rule_id_or_section"]}],',
    '  "key_factors": [{"frame_id": "frame_00x", "timestamp": "M:SS.ss", "factor": "string"}],',
    '  "rule_basis": "string",',
    '  "relevant_rules_used": [{"issue_id":"issue_001","rule_id":"string","section":"string","why_relevant":"string"}],',
    '  "limitations": ["string"]',
    "}",
]

_VERDICT_LABELS = {"Fair Call", "Bad Call", "Inconclusive"}
_VERDICT_CONFIDENCE_MAP = {"low": "Low", "medium": "Medium", "high": "High"}


def _normalize_verdict_label(value) -> str:
    candidate = (str(value or "")).strip().lower()
    mapping = {
        "fair call": "Fair Call",
        "fair": "Fair Call",
        "bad call": "Bad Call",
        "bad": "Bad Call",
        "inconclusive": "Inconclusive",
        "unclear": "Inconclusive",
    }
    return mapping.get(candidate, "Inconclusive")


def _normalize_verdict_confidence(value) -> str:
    return _VERDICT_CONFIDENCE_MAP.get((str(value or "")).strip().lower(), "Low")


def _has_contact_quality_limitations(visual_result: dict) -> bool:
    text = " ".join(
        str(item)
        for item in [
            *(visual_result.get("limitations") or []),
            *(visual_result.get("missing_evidence") or []),
            visual_result.get("high_contact_description", ""),
        ]
    ).lower()
    return any(term in text for term in ["blur", "occlusion", "occluded", "partial", "unclear", "cropped"])


def _normalize_verdict_factors(payload_factors, frame_id_map: dict) -> list[dict]:
    if not isinstance(payload_factors, list):
        return []
    normalized = []
    for item in payload_factors[:6]:
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("frame_id", "")).strip()
        if frame_id not in frame_id_map:
            continue
        factor = _complete_sentence(str(item.get("factor", "")).strip())
        if not factor:
            continue
        timestamp = str(item.get("timestamp", "")).strip() or frame_id_map[frame_id].get("timestamp", "")
        normalized.append(
            {"frame_id": frame_id, "timestamp": timestamp[:20], "factor": factor[:220]}
        )
    return normalized


def _normalize_issue_analysis(payload_items, frame_id_map: dict, matched_rules_by_issue: dict) -> list[dict]:
    if not isinstance(payload_items, list):
        return []
    known_rule_ids = {
        str(rule.get("rule_id") or rule.get("section") or "")
        for rules in (matched_rules_by_issue or {}).values()
        for rule in rules
    }
    normalized = []
    for index, item in enumerate(payload_items[:10], start=1):
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get("issue_id") or f"issue_{index:03d}").strip()
        frame_ids = []
        for frame_id in item.get("relevant_frames") or []:
            candidate = str(frame_id).strip()
            if candidate in frame_id_map and candidate not in frame_ids:
                frame_ids.append(candidate)
        relevant_rules = []
        for rule_id in item.get("relevant_rules") or []:
            candidate = str(rule_id).strip()
            if candidate and (not known_rule_ids or candidate in known_rule_ids) and candidate not in relevant_rules:
                relevant_rules.append(candidate[:120])
        finding = _complete_sentence(str(item.get("finding", "")).strip(), max_len=500)
        if not finding:
            continue
        normalized.append(
            {
                "issue_id": issue_id,
                "issue_type": _normalize_issue_type(item.get("issue_type", "")),
                "finding": finding,
                "confidence": _normalize_confidence(item.get("confidence", "")),
                "relevant_frames": frame_ids,
                "relevant_rules": relevant_rules,
            }
        )
    return normalized


def _rule_lookup_by_key(matched_rules_by_issue: dict) -> dict[str, tuple[str, dict]]:
    lookup = {}
    for issue_id, rules in (matched_rules_by_issue or {}).items():
        for rule in rules:
            for key in [str(rule.get("rule_id") or "").strip(), str(rule.get("section") or "").strip()]:
                if key and key not in lookup:
                    lookup[key] = (str(issue_id), rule)
    return lookup


def _normalize_relevant_rules_used(payload_items, matched_rules_by_issue: dict, issue_analysis: list[dict]) -> list[dict]:
    lookup = _rule_lookup_by_key(matched_rules_by_issue)
    normalized = []
    seen = set()

    def add(issue_id: str, rule: dict, why_relevant: str) -> None:
        key = str(rule.get("rule_id") or rule.get("section") or "").strip()
        if not key or key in seen or len(normalized) >= 6:
            return
        seen.add(key)
        normalized.append(
            {
                "issue_id": str(issue_id or "").strip()[:120],
                "rule_id": str(rule.get("rule_id") or "").strip()[:120],
                "section": str(rule.get("section") or "").strip()[:120],
                "why_relevant": _complete_sentence(str(why_relevant or "").strip(), max_len=350)
                or "Retrieved rulebook context used by the Verdict Agent.",
            }
        )

    if isinstance(payload_items, list):
        for item in payload_items[:8]:
            if not isinstance(item, dict):
                continue
            rule_key = str(item.get("rule_id") or item.get("section") or "").strip()
            if rule_key not in lookup:
                continue
            issue_id, rule = lookup[rule_key]
            add(item.get("issue_id") or issue_id, rule, item.get("why_relevant", ""))

    if normalized:
        return normalized

    for issue in issue_analysis or []:
        for rule_key in issue.get("relevant_rules") or []:
            if rule_key not in lookup:
                continue
            issue_id, rule = lookup[rule_key]
            add(
                issue.get("issue_id") or issue_id,
                rule,
                f"Relevant to {issue.get('issue_type', 'the reviewed issue')} analysis.",
            )

    if normalized:
        return normalized

    for issue_id, rules in (matched_rules_by_issue or {}).items():
        for rule in rules[:1]:
            add(issue_id, rule, "Top retrieved rulebook match for this issue.")
    return normalized


def _normalize_other_possible_issues(payload_items) -> list[dict]:
    if not isinstance(payload_items, list):
        return []
    normalized = []
    for item in payload_items[:6]:
        if not isinstance(item, dict):
            continue
        issue = _complete_sentence(str(item.get("issue", "")).strip(), max_len=220)
        reason = _complete_sentence(str(item.get("reason", "")).strip(), max_len=500)
        if not issue and not reason:
            continue
        normalized.append(
            {
                "issue": issue or "Possible secondary issue.",
                "confidence": _normalize_role_confidence(item.get("confidence", "low"), default="low"),
                "reason": reason or "Secondary observation from available frames.",
            }
        )
    return normalized


def _new_verdict_result(model_name: str) -> dict:
    return {
        "agent": "verdict_agent",
        "success": False,
        "status": "Skipped",
        "skipped_reason": "",
        "original_call": "",
        "verdict": "Inconclusive",
        "confidence": "Low",
        "correct_ruling_if_any": "",
        "reasoning": "",
        "selected_call_reasoning": "",
        "other_possible_issues": [],
        "issue_analysis": [],
        "key_factors": [],
        "rule_basis": "",
        "relevant_rules_used": [],
        "matched_rules": [],
        "matched_rules_by_issue": {},
        "limitations": [],
        "model_used": model_name,
        "error": None,
        "error_type": None,
        "debug": {
            "model_used": model_name,
            "primary_model_attempted": model_name,
            "fallback_model_attempted": _gemini_fallback_model(),
            "final_model_used": None,
            "retry_count": 0,
            "fallback_used": False,
            "primary_model_error_message": "",
            "frames_sent_to_gemini_count": 0,
            "gemini_api_call_succeeded": False,
            "response_parsing_succeeded": False,
            "raw_response_excerpt": "",
            "error_type": None,
            "error_message": None,
        },
    }


def _cap_frames_with_phase_spread(frames: list[dict], max_frames: int) -> list[dict]:
    if len(frames) <= max_frames:
        return sorted(frames, key=lambda frame: _timestamp_to_seconds(frame.get("timestamp", "")))
    selected = []
    seen = set()
    phase_order = [
        "likely_contact",
        "pre_contact",
        "post_contact",
        "clip_start_context",
        "clip_end_context",
        "aftermath",
        "temporal_coverage",
    ]

    def add(frame: dict) -> bool:
        frame_id = frame.get("frame_id")
        if not frame_id or frame_id in seen or len(selected) >= max_frames:
            return False
        selected.append(frame)
        seen.add(frame_id)
        return True

    for phase in phase_order:
        phase_frames = [
            frame for frame in frames
            if (frame.get("phase_bucket") or _selection_phase(frame.get("reason", ""))) == phase
        ]
        if not phase_frames:
            continue
        add(max(phase_frames, key=lambda frame: (frame.get("weight", 0), frame.get("score", 0.0))))

    for frame in sorted(frames, key=lambda item: (item.get("weight", 0), item.get("score", 0.0)), reverse=True):
        if len(selected) >= max_frames:
            break
        add(frame)

    return sorted(selected, key=lambda frame: _timestamp_to_seconds(frame.get("timestamp", "")))


def _pick_verdict_frames(visual_result: dict) -> list[dict]:
    evidence_frames = visual_result.get("evidence_frames") or []
    recommended = visual_result.get("recommended_frames_for_verdict_agent") or []
    by_id = {f["frame_id"]: f for f in evidence_frames if isinstance(f, dict) and f.get("frame_id")}
    max_frames = min(_refcheck_max_verdict_frames(), _refcheck_max_gemini_images())
    goaltending_focus = _is_goaltending_call(visual_result.get("original_call")) or visual_result.get("primary_issue") == "goaltending"

    selected: list[tuple[int, dict]] = []
    seen: set[str] = set()

    def add_frame(frame_id: str, reason: str, weight: int) -> None:
        frame_id = str(frame_id or "").strip()
        if frame_id in seen or frame_id not in by_id:
            return
        base = by_id[frame_id]
        if not base.get("local_path"):
            return
        seen.add(frame_id)
        selected.append(
            (
                weight,
                {
                    "frame_id": frame_id,
                    "timestamp": base.get("timestamp", ""),
                    "local_path": base["local_path"],
                    "reason": (reason or base.get("selection_reason", "selected for temporal coverage"))[:220],
                    "score": base.get("score", 0.0),
                    "weight": weight,
                    "phase_bucket": base.get("phase_bucket") or _selection_phase(base.get("selection_reason", "")),
                },
            )
        )

    if goaltending_focus:
        for rec in recommended:
            if not isinstance(rec, dict):
                continue
            reason_text = str(rec.get("reason", "")).strip().lower()
            reason_weight = 98 if any(
                marker in reason_text for marker in ["ball", "rim", "cylinder", "backboard", "goaltend", "trajectory"]
            ) else 86
            add_frame(rec.get("frame_id", ""), str(rec.get("reason", "")).strip(), reason_weight)

        for issue in visual_result.get("issue_cards") or []:
            if not isinstance(issue, dict):
                continue
            issue_type = issue.get("issue_type")
            issue_weight = 96 if issue_type == "goaltending" else 65
            for frame_id in issue.get("frame_ids") or []:
                add_frame(frame_id, issue.get("description", "issue evidence"), issue_weight)

        for frame_id in visual_result.get("high_contact_frame_ids") or []:
            add_frame(frame_id, "secondary contact context", 58)

        for contact in visual_result.get("contact_points") or []:
            if not isinstance(contact, dict):
                continue
            add_frame(contact.get("frame_id", ""), contact.get("description", "contact point"), 55)
    else:
        for frame_id in visual_result.get("high_contact_frame_ids") or []:
            add_frame(frame_id, "high contact frame", 100)

        for contact in visual_result.get("contact_points") or []:
            if not isinstance(contact, dict):
                continue
            target = contact.get("target_body_area")
            weight = 95 if target in {"face", "head", "neck"} else 80
            add_frame(contact.get("frame_id", ""), contact.get("description", "contact point"), weight)

        for issue in visual_result.get("issue_cards") or []:
            if not isinstance(issue, dict):
                continue
            issue_weight = 90 if issue.get("issue_type") == "high_contact" else 70
            for frame_id in issue.get("frame_ids") or []:
                add_frame(frame_id, issue.get("description", "issue evidence"), issue_weight)

        for rec in recommended:
            if not isinstance(rec, dict):
                continue
            add_frame(rec.get("frame_id", ""), str(rec.get("reason", "")).strip(), 65)

    if selected:
        selected_frames = [
            item[1]
            for item in sorted(selected, key=lambda item: (item[0], item[1].get("score", 0.0)), reverse=True)
        ]
        return _cap_frames_with_phase_spread(selected_frames, max_frames)

    fallback = []
    for base in sorted(evidence_frames, key=lambda frame: frame.get("score", 0.0), reverse=True)[:max_frames]:
        if not isinstance(base, dict) or not base.get("local_path"):
            continue
        fallback.append(
            {
                "frame_id": base.get("frame_id", ""),
                "timestamp": base.get("timestamp", ""),
                "local_path": base["local_path"],
                "reason": base.get("selection_reason", "selected for temporal coverage"),
                "phase_bucket": base.get("phase_bucket") or _selection_phase(base.get("selection_reason", "")),
                "weight": 0,
                "score": base.get("score", 0.0),
            }
        )
    return _cap_frames_with_phase_spread(fallback, max_frames)


def _build_verdict_prompt(
    video_summary: str,
    frames_for_verdict: list[dict],
    original_call: str | None = None,
    advisory: dict | None = None,
    matched_rules: list[dict] | None = None,
    matched_rules_by_issue: dict | None = None,
    officiating_issue_summary: str | None = None,
    key_events: list[dict] | None = None,
    possible_critical_moments: list[dict] | None = None,
    issue_cards: list[dict] | None = None,
    contact_points: list[dict] | None = None,
    primary_issue: str | None = None,
    secondary_issues: list[dict] | None = None,
    role_context: dict | None = None,
) -> str:
    lines = list(_VERDICT_PROMPT_INSTRUCTIONS)
    lines.append("")
    lines.append("Visual Analyst summary (verbatim):")
    lines.append('"""')
    lines.append(video_summary.strip())
    lines.append('"""')
    lines.append("")
    if officiating_issue_summary:
        lines.append("Agent 1 neutral officiating issue summary:")
        lines.append(str(officiating_issue_summary).strip())
        lines.append("")
    lines.append(f'Original call: "{(original_call or "").strip() or "Unknown"}"')
    if primary_issue:
        lines.append(f'Primary issue from Agent 1: "{primary_issue}"')
    if key_events:
        lines.append("")
        lines.append("Frame-bound key events from Agent 1:")
        for event in key_events[:10]:
            if not isinstance(event, dict):
                continue
            lines.append(
                f'- {event.get("frame_id", "")} at {event.get("timestamp", "")}: '
                f'{event.get("description", "")}'
            )
    if possible_critical_moments:
        lines.append("")
        lines.append("Possible critical moments from Agent 1:")
        for moment in possible_critical_moments[:10]:
            if not isinstance(moment, dict):
                continue
            lines.append(
                f'- {moment.get("frame_id", "")} at {moment.get("timestamp", "")}: '
                f'{moment.get("description", "")} Why relevant: {moment.get("why_relevant", "")}.'
            )
    if role_context:
        lines.append("")
        lines.append("Role context from Agent 1:")
        lines.append(
            f'- offense_team_color: {role_context.get("offense_team_color", "unknown")} | defense_team_color: {role_context.get("defense_team_color", "unknown")}'
        )
        lines.append(
            f'- shooter_or_finisher: {role_context.get("shooter_or_finisher", "unknown")} | primary_contesting_defender: {role_context.get("primary_contesting_defender", "unknown")}'
        )
        lines.append(f'- role_confidence: {role_context.get("role_confidence", "low")}')
        role_uncertainty = role_context.get("role_uncertainty_reason")
        if role_uncertainty:
            lines.append(f"- role_uncertainty_reason: {role_uncertainty}")
        contact_direction = role_context.get("contact_direction_assessment")
        if isinstance(contact_direction, dict) and contact_direction:
            lines.append(
                "- contact_direction_assessment: "
                f"initiator={contact_direction.get('contact_initiator', 'unknown')}; "
                f"recipient={contact_direction.get('contact_recipient', 'unknown')}; "
                f"location={contact_direction.get('contact_location', 'unknown')}; "
                f"context={contact_direction.get('contact_context', 'unknown')}; "
                f"effect={contact_direction.get('contact_effect', 'unknown')}; "
                f"confidence={contact_direction.get('contact_confidence', 'low')}"
            )
    lines.append("")
    if issue_cards:
        lines.append("Issue cards from Agent 1:")
        for issue in issue_cards[:10]:
            lines.append(
                f'- {issue.get("issue_id", "issue")} ({issue.get("issue_type", "unclear")}): '
                f'{issue.get("description", "")} Frames: {", ".join(issue.get("frame_ids") or [])}.'
            )
    if contact_points:
        lines.append("")
        lines.append("Contact points from Agent 1:")
        for contact in contact_points[:12]:
            lines.append(
                f'- {contact.get("frame_id", "")} at {contact.get("timestamp", "")}: '
                f'{contact.get("contact_type", "unclear")} by {contact.get("actor", "unclear")} '
                f'toward {contact.get("target_body_area", "unclear")} '
                f'({contact.get("confidence", "Low")}): {contact.get("description", "")}'
            )
    if secondary_issues:
        lines.append("")
        lines.append("Secondary issues from Agent 1 (do not override selected call verdict):")
        for item in secondary_issues[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f'- {item.get("issue", "secondary issue")} ({item.get("confidence", "low")}): {item.get("reason", "")}'
            )
    lines.append("")
    lines.append("Curated frames for verdict review (in temporal order):")
    for frame in frames_for_verdict:
        lines.append(
            f'- {frame["frame_id"]} at {frame["timestamp"]} -> {frame.get("reason", "context")}'
        )
    if matched_rules_by_issue:
        lines.append("")
        lines.append("Rulebook context grouped by issue:")
        for issue_id, rules in list(matched_rules_by_issue.items())[:12]:
            if not rules:
                continue
            lines.append(f"- {issue_id}:")
            for rule in rules[:4]:
                section = rule.get("section") or rule.get("rule_id") or "rule"
                text = (rule.get("text") or "").strip()
                if text:
                    lines.append(f"  * [{section}] {text}")
                else:
                    lines.append(f"  * [{section}] (no text available)")
    if matched_rules:
        lines.append("")
        lines.append(
            "Rulebook context (top matches retrieved from the NBA rulebook via semantic search on the visual summary; "
            "treat these as the primary basis for your rule_basis field — cite the most relevant one and stay neutral):"
        )
        for rule in matched_rules[:6]:
            section = rule.get("section") or rule.get("rule_id") or "rule"
            text = (rule.get("text") or "").strip()
            if text:
                lines.append(f"- [{section}] {text}")
            else:
                lines.append(f"- [{section}] (no text available)")
    if advisory:
        lines.append("")
        lines.append("Agent 1 advisory signals (informational only; not a gate on your verdict):")
        readiness = advisory.get("verdict_readiness")
        if readiness:
            lines.append(f"- verdict_readiness: {readiness}")
        if "can_reason_about_call" in advisory:
            lines.append(f"- can_reason_about_call: {bool(advisory.get('can_reason_about_call'))}")
        sequence = advisory.get("sequence_interpretation")
        if sequence:
            lines.append(f"- sequence_interpretation: {sequence}")
        visual_confidence = advisory.get("confidence_in_visual_description")
        if visual_confidence:
            lines.append(f"- confidence_in_visual_description: {visual_confidence}")
        if "possible_high_contact" in advisory:
            lines.append(f"- possible_high_contact: {bool(advisory.get('possible_high_contact'))}")
        high_contact_description = advisory.get("high_contact_description")
        if high_contact_description:
            lines.append(f"- high_contact_description: {high_contact_description}")
        missing = advisory.get("missing_evidence") or []
        if missing:
            lines.append("- missing_evidence (lean toward Inconclusive when these block the call):")
            for item in missing[:6]:
                lines.append(f"  * {item}")
    return "\n".join(lines)


def _call_gemini_for_verdict(
    api_key: str,
    model_name: str,
    prompt: str,
    frames_for_verdict: list[dict],
    result: dict,
) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    frames_for_verdict = frames_for_verdict[: min(_refcheck_max_verdict_frames(), _refcheck_max_gemini_images())]
    uploaded_parts = [client.files.upload(file=frame["local_path"]) for frame in frames_for_verdict]
    result["debug"]["frames_sent_to_gemini_count"] = len(uploaded_parts)
    raw_text = _generate_gemini_with_fallback(
        client=client,
        primary_model=model_name,
        fallback_model=_gemini_fallback_model(),
        contents=[prompt, *uploaded_parts],
        debug=result["debug"],
    )
    result["debug"]["gemini_api_call_succeeded"] = True
    final_model = result["debug"].get("final_model_used") or model_name
    result["model_used"] = final_model
    result["debug"]["model_used"] = final_model
    result["debug"]["raw_response_excerpt"] = _safe_excerpt(raw_text)
    return raw_text


def _apply_verdict_skipped(result: dict, reason: str) -> None:
    result["status"] = "Skipped"
    result["success"] = False
    result["skipped_reason"] = reason


def _apply_verdict_missing_api_key(result: dict) -> None:
    result["status"] = "Failed"
    result["error_type"] = "missing_api_key"
    result["error"] = "GEMINI_API_KEY is not configured. Verdict Agent could not run."
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = result["error"]


def _apply_verdict_empty_response(result: dict) -> None:
    result["status"] = "Failed"
    result["error_type"] = "empty_response"
    result["error"] = "Verdict Agent returned an empty response."
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = result["error"]


def _apply_verdict_unstructured_response(result: dict, raw_text: str) -> None:
    result["status"] = "Failed"
    result["error_type"] = "json_parsing_failure"
    result["error"] = "Verdict Agent returned unstructured text."
    result["reasoning"] = _strip_code_fences(raw_text)[:600]
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = result["error"]


def _apply_verdict_json_decode_failure(result: dict, raw_text: str, exc: Exception) -> None:
    result["status"] = "Failed"
    result["error_type"] = "json_parsing_failure"
    result["error"] = "Verdict Agent JSON could not be parsed."
    result["reasoning"] = _strip_code_fences(raw_text)[:600]
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = str(exc)[:300]


def _apply_verdict_api_failure(result: dict, exc: Exception) -> None:
    result["status"] = "Failed"
    result["error_type"] = "gemini_api_failure"
    result["error"] = "Verdict Agent could not be completed for this upload."
    result["debug"]["error_type"] = result["error_type"]
    result["debug"]["error_message"] = str(exc)[:300]


def _apply_verdict_payload(
    result: dict,
    payload: dict,
    frame_id_map: dict,
    matched_rules_by_issue: dict,
    original_call: str,
    visual_result: dict,
) -> None:
    goaltending_selected = _is_goaltending_call(original_call)
    result["agent"] = str(payload.get("agent", "verdict_agent")).strip().lower() or "verdict_agent"
    result["original_call"] = original_call or ""
    result["verdict"] = _normalize_verdict_label(payload.get("verdict"))
    result["confidence"] = _normalize_verdict_confidence(payload.get("confidence"))
    if not original_call:
        result["verdict"] = "Inconclusive"
    if visual_result.get("confidence_in_visual_description") == "Low" and result["confidence"] == "High":
        result["confidence"] = "Medium"
    if visual_result.get("possible_high_contact") and result["verdict"] == "Fair Call" and result["confidence"] == "High":
        result["confidence"] = "Medium"
    if _has_contact_quality_limitations(visual_result) and result["confidence"] == "High":
        result["confidence"] = "Medium"
    result["correct_ruling_if_any"] = str(payload.get("correct_ruling_if_any", "")).strip()[:300]
    result["reasoning"] = str(payload.get("reasoning", "")).strip()[:1200]
    result["selected_call_reasoning"] = str(payload.get("selected_call_reasoning", "")).strip()[:1200]
    result["other_possible_issues"] = _normalize_other_possible_issues(payload.get("other_possible_issues"))
    if not result["other_possible_issues"]:
        result["other_possible_issues"] = _normalize_other_possible_issues(visual_result.get("secondary_issues"))
    result["issue_analysis"] = _normalize_issue_analysis(
        payload.get("issue_analysis"), frame_id_map, matched_rules_by_issue
    )
    result["key_factors"] = _normalize_verdict_factors(payload.get("key_factors"), frame_id_map)
    result["rule_basis"] = str(payload.get("rule_basis", "")).strip()[:500]
    result["relevant_rules_used"] = _normalize_relevant_rules_used(
        payload.get("relevant_rules_used"), matched_rules_by_issue, result["issue_analysis"]
    )
    if result["relevant_rules_used"] and not result["rule_basis"]:
        first_rule = result["relevant_rules_used"][0]
        rule_name = first_rule.get("section") or first_rule.get("rule_id") or "retrieved rule"
        result["rule_basis"] = f"{rule_name}: {first_rule.get('why_relevant', '')}"[:500]
    raw_limitations = payload.get("limitations") or []
    result["limitations"] = [
        str(item).strip()[:220] for item in raw_limitations[:6] if str(item).strip()
    ]

    if goaltending_selected:
        if not result["selected_call_reasoning"]:
            result["selected_call_reasoning"] = result["reasoning"]
        goaltending_reasoning_text = " ".join(
            [
                result.get("selected_call_reasoning", ""),
                result.get("reasoning", ""),
                " ".join(visual_result.get("missing_evidence") or []),
                " ".join(result.get("limitations") or []),
            ]
        ).lower()
        required_markers = ["downward flight", "backboard", "cylinder", "ball trajectory", "defender-ball", "defender ball"]
        has_goaltending_basis = any(marker in goaltending_reasoning_text for marker in required_markers)
        if not has_goaltending_basis or not visual_result.get("can_reason_about_call", False):
            result["verdict"] = "Inconclusive"
            if result["confidence"] == "High":
                result["confidence"] = "Medium"

    result["status"] = "Analyzed"
    result["success"] = True
    result["error"] = None
    result["error_type"] = None


def run_verdict_agent(visual_result: dict) -> dict:
    model_name = _gemini_verdict_model()
    result = _new_verdict_result(model_name)
    original_call = _normalize_original_call(visual_result.get("original_call"))
    result["original_call"] = original_call

    if visual_result.get("status") != "Analyzed":
        _apply_verdict_skipped(result, "Agent 1 did not produce an Analyzed result.")
        return result
    video_summary = (visual_result.get("video_summary") or "").strip()
    if not video_summary:
        _apply_verdict_skipped(result, "No video_summary available.")
        return result

    matched_rules_by_issue = _cap_grouped_rules_total(retrieve_rules_for_issues(visual_result), max_rules=3)
    matched_rules = _dedupe_rules(
        [rule for rules in matched_rules_by_issue.values() for rule in rules]
    )[:3]
    result["matched_rules"] = matched_rules
    result["matched_rules_by_issue"] = matched_rules_by_issue

    frames = _pick_verdict_frames(visual_result)
    if not frames:
        _apply_verdict_skipped(result, "No verdict frames available.")
        return result

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _apply_verdict_missing_api_key(result)
        return result

    advisory = {
        "verdict_readiness": visual_result.get("verdict_readiness"),
        "can_reason_about_call": visual_result.get("can_reason_about_call"),
        "sequence_interpretation": visual_result.get("sequence_interpretation"),
        "confidence_in_visual_description": visual_result.get("confidence_in_visual_description"),
        "possible_high_contact": visual_result.get("possible_high_contact"),
        "high_contact_description": visual_result.get("high_contact_description"),
        "missing_evidence": visual_result.get("missing_evidence") or [],
    }
    prompt = _build_verdict_prompt(
        video_summary,
        frames,
        original_call=original_call,
        advisory=advisory,
        matched_rules=matched_rules,
        matched_rules_by_issue=matched_rules_by_issue,
        officiating_issue_summary=visual_result.get("officiating_issue_summary"),
        key_events=visual_result.get("key_events") or [],
        possible_critical_moments=visual_result.get("possible_critical_moments") or [],
        issue_cards=visual_result.get("issue_cards") or [],
        contact_points=visual_result.get("contact_points") or [],
        primary_issue=visual_result.get("primary_issue"),
        secondary_issues=visual_result.get("secondary_issues") or [],
        role_context={
            "offense_team_color": visual_result.get("offense_team_color", "unknown"),
            "defense_team_color": visual_result.get("defense_team_color", "unknown"),
            "shooter_or_finisher": visual_result.get("shooter_or_finisher", "unknown"),
            "primary_contesting_defender": visual_result.get("primary_contesting_defender", "unknown"),
            "role_confidence": visual_result.get("role_confidence", "low"),
            "role_uncertainty_reason": visual_result.get("role_uncertainty_reason", ""),
            "contact_direction_assessment": visual_result.get("contact_direction_assessment") or {},
        },
    )
    raw_text = ""
    try:
        raw_text = _call_gemini_for_verdict(api_key, model_name, prompt, frames, result)

        if not raw_text:
            _apply_verdict_empty_response(result)
            return result

        json_candidate = _extract_json_object(raw_text)
        if not json_candidate:
            _apply_verdict_unstructured_response(result, raw_text)
            return result

        payload = json.loads(json_candidate)
        result["debug"]["response_parsing_succeeded"] = True
        frame_id_map = {f["frame_id"]: f for f in frames}
        _apply_verdict_payload(result, payload, frame_id_map, matched_rules_by_issue, original_call, visual_result)
    except json.JSONDecodeError as exc:
        _apply_verdict_json_decode_failure(result, raw_text, exc)
    except Exception as exc:
        _apply_verdict_api_failure(result, exc)

    return result


def analyze_video_with_gemini(video_path: str, original_call: str | None = None) -> dict:
    model_name = _gemini_visual_model()
    result = _new_visual_result(model_name)
    result["original_call"] = _normalize_original_call(original_call)

    cv2_mod, capture = _open_video(video_path, result)
    if capture is None:
        return result

    try:
        fps, total_frames, metadata = _read_video_metadata(capture, cv2_mod)
        result["metadata"] = metadata

        if total_frames <= 0:
            _set_failure(result, "frame_extraction_failure", "No readable frames were found in the uploaded video.")
            return result

        selected_candidates, selection_mode = _select_candidate_frames(capture, total_frames, fps, result)
    finally:
        capture.release()

    if not selected_candidates:
        _set_failure(result, "frame_extraction_failure", "Could not extract representative frames from this video.")
        return result

    frames_dir = settings.MEDIA_ROOT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    evidence_frames = _persist_evidence_frames(selected_candidates, frames_dir, cv2_mod)

    if not evidence_frames:
        _set_failure(result, "frame_extraction_failure", "Could not save selected evidence frames.")
        return result

    result["evidence_frames"] = evidence_frames
    result["status"] = "Partial"
    result["debug"]["frames_extracted_count"] = len(evidence_frames)
    result["debug"]["selected_evidence_count"] = len(evidence_frames)
    result["recommended_frames_for_verdict_agent"] = _default_recommended_frames(evidence_frames)
    result["evidence_selection_summary"] = _build_selection_summary(
        selection_mode,
        result["debug"]["candidates_scanned_count"],
        len(evidence_frames),
        result["debug"],
    )
    result["possible_critical_moments"] = _initial_critical_moments(evidence_frames)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _apply_missing_api_key(result)
        _clean_for_session(result)
        return result

    _run_gemini_analysis(api_key, model_name, evidence_frames, original_call, result)
    if result.get("status") == "Analyzed":
        result["verdict"] = run_verdict_agent(result)
    else:
        result["verdict"] = _new_verdict_result(_gemini_verdict_model())
        _apply_verdict_skipped(result["verdict"], "Agent 1 failed; verdict stage was not executed.")
    _clean_for_session(result)
    return result
