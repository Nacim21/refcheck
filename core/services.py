import json
import os
from pathlib import Path
from uuid import uuid4

from django.conf import settings


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
    allowed = {"single_play", "multiple_plays", "possible_replay", "unclear"}
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


def _candidate_indices(total_frames: int, fps: float, duration: float) -> list[int]:
    scan_fps = 2.0 if duration <= 20 else 1.5
    step = max(1, int(round(fps / scan_fps))) if fps > 0 else 1
    start_frame = int(round((total_frames - 1) * 0.05))
    end_frame = int(round((total_frames - 1) * 0.95))
    indices = list(range(start_frame, max(start_frame + 1, end_frame + 1), step))
    if not indices:
        indices = [max(0, total_frames // 2)]
    max_candidates = 30
    if len(indices) > max_candidates:
        sampled = []
        for i in range(max_candidates):
            pos = int(round(i * (len(indices) - 1) / (max_candidates - 1)))
            sampled.append(indices[pos])
        indices = sampled
    return sorted(set(indices))


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
        if prev_small_gray is not None:
            diff = cv2.absdiff(gray, prev_small_gray)
            motion = float(diff.mean() / 255.0)

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

        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        duplicate_similarity = 0.0
        if prev_hist is not None:
            duplicate_similarity = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL))
        duplicate_similarity = max(-1.0, min(1.0, duplicate_similarity))

        timestamp_seconds = (frame_idx / fps) if fps > 0 else 0.0
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
                "selection_reason": "selected for temporal coverage",
            }
        )

        prev_small_gray = gray
        prev_hist = hist
    return candidates


def _select_evidence_candidates(candidates: list[dict], target_min: int = 8, target_max: int = 12):
    if not candidates:
        return []

    motions = [c["motion"] for c in candidates]
    sharpnesses = [c["sharpness"] for c in candidates]
    relevances = [c["relevance"] for c in candidates]

    motion_norm = _safe_minmax(motions)
    sharp_norm = _safe_minmax(sharpnesses)
    relevance_norm = _safe_minmax(relevances)

    for i, candidate in enumerate(candidates):
        duplicate_penalty = max(0.0, candidate["duplicate_similarity"])
        uniqueness = 1.0 - duplicate_penalty
        score = (
            (0.50 * motion_norm[i])
            + (0.20 * sharp_norm[i])
            + (0.20 * relevance_norm[i])
            + (0.10 * uniqueness)
        )
        candidate["score"] = float(score)
        candidate["motion_norm"] = float(motion_norm[i])

    ranked_motion = sorted(candidates, key=lambda c: c["motion_norm"], reverse=True)
    top_motion = ranked_motion[:3]
    selected_ids = set()
    selected = []

    for peak in top_motion:
        peak_pos = peak["candidate_id"]
        for offset, reason in [(-1, "pre-contact context"), (0, "high motion"), (1, "post-contact context")]:
            neighbor_pos = peak_pos + offset
            if neighbor_pos < 0 or neighbor_pos >= len(candidates):
                continue
            item = candidates[neighbor_pos]
            if item["candidate_id"] in selected_ids:
                continue
            item["selection_reason"] = reason
            selected.append(item)
            selected_ids.add(item["candidate_id"])
            if len(selected) >= target_max:
                break
        if len(selected) >= target_max:
            break

    ranked_score = sorted(candidates, key=lambda c: c["score"], reverse=True)
    for item in ranked_score:
        if len(selected) >= target_max:
            break
        if item["candidate_id"] in selected_ids:
            continue
        too_close = any(abs(item["candidate_id"] - existing["candidate_id"]) <= 1 for existing in selected)
        if too_close and len(selected) >= target_min:
            continue
        if item["selection_reason"] == "selected for temporal coverage":
            if item["motion_norm"] >= 0.55:
                item["selection_reason"] = "high motion"
            elif item["motion_norm"] <= 0.20:
                item["selection_reason"] = "selected for temporal coverage"
            else:
                item["selection_reason"] = "possible contact context"
        selected.append(item)
        selected_ids.add(item["candidate_id"])

    selected = sorted(selected, key=lambda c: c["timestamp_seconds"])
    if len(selected) > target_max:
        selected = selected[:target_max]

    if len(selected) < target_min:
        for item in sorted(candidates, key=lambda c: c["timestamp_seconds"]):
            if item["candidate_id"] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item["candidate_id"])
            if len(selected) >= target_min:
                break
        selected = sorted(selected, key=lambda c: c["timestamp_seconds"])
    return selected


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
        if "high motion" in reason:
            weight = 3
        elif "pre-contact" in reason or "post-contact" in reason:
            weight = 2
        elif "possible contact" in reason:
            weight = 1
        priority.append((weight, frame))
    priority.sort(key=lambda item: item[0], reverse=True)
    top = [item[1] for item in priority[:5]]
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
    "# WHAT TO DESCRIBE (BE EXHAUSTIVE, DO NOT LEAVE OUT DETAILS)",
    "Describe every visually relevant element. Aim for a thorough, evidence-grade description rather than a short summary.",
    "For each frame and across the sequence, cover at minimum:",
    "- Setting and context: court or field type if visible, visible court markings (three-point line, free-throw line, restricted area arc, sideline, baseline, midcourt line, key/paint), scoreboard or clock overlays, broadcast graphics, on-screen text, score, time remaining, and any visible 'REPLAY' or angle labels.",
    "- Camera: framing (wide, medium, tight), apparent angle (baseline, sideline, overhead, behind-the-basket), zoom level, and whether the angle changes between frames.",
    "- Players involved: jersey color, jersey number if legible, approximate role (ball handler, defender, shooter, screener, off-ball), and stance. Refer to players consistently across frames (e.g., 'white #23', 'dark-jersey defender').",
    "- Body positions and biomechanics: feet placement (set, sliding, airborne, on heels, on toes), torso lean, hip orientation, arm/hand position, head direction, point of balance, and whether the player appears stationary or moving.",
    "- Ball state: who possesses it, height (floor, waist, chest, above head), whether it is being dribbled, gathered, released, in flight, deflected, or loose; visible spin or trajectory if discernible.",
    "- Contact: where on the body contact appears to occur (chest-to-chest, hip, shoulder, arm, hand, leg, head), which player initiates visible contact relative to the frame shown, and the visible reaction (recoil, fall, no reaction).",
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
    "# REPLAY VS SEPARATE PLAY",
    "Distinguish between a genuinely separate incident and the same incident shown again as a broadcast replay or alternate camera angle.",
    "Replay indicators include: same players in matching jerseys repeating the same action, an obvious camera-angle change with similar body poses, slow-motion appearance, on-screen 'REPLAY' text, repeated scoreboard state, or a graphic wipe between sequences.",
    "If you see any such indicators and no clear evidence of a distinct play, set sequence_interpretation to 'possible_replay' and possible_replay_duplicate to true.",
    "Do not describe replay-like sequences as separate incidents unless there is clear visual evidence (e.g., different score, different players, different court location) that they are separate plays.",
    "",
    "# CALL TYPE FIELDS",
    "Do not infer visible_call_type from player actions. Player movement alone never establishes a call type.",
    "Set visible_call_type to 'unclear' unless the call type is explicitly visible as on-screen text/overlay/graphic, or has been provided by the user as the original call.",
    "If visible_call_type uses the user-provided original call, visible_call_type_reason must explicitly state that the value is user-provided and is not a visual or rules conclusion drawn by you.",
    "visible_call_type_reason must either (a) cite the exact visible basis (e.g., 'overlay text reads BLOCK on frame_004') or (b) state that Agent 1 is not making a call classification.",
    "possible_call_types may list visual categories a later Verdict Agent might consider, but this is suggestive, not a final decision.",
    "officiating_issue_summary must be a neutral, visually grounded summary of what a future Verdict Agent may need to examine. It is not a rules conclusion and must not contain a verdict.",
    "",
    "# QUALITY AND READINESS FIELDS",
    "visual_frame_quality rates image clarity and usefulness only (lighting, focus, motion blur, occlusion, resolution, framing). It does not rate the play itself.",
    "verdict_readiness rates whether a future Verdict Agent has enough visual continuity (before / during / after the key moment) to reason about the call.",
    "can_reason_about_call indicates only whether selected frames contain enough visual continuity for a later Verdict Agent to reason. It is not your decision on the call.",
    "Set can_reason_about_call to false if any of the following hold: defender feet are not clearly visible before contact, restricted-area or boundary context is unclear, the exact moment of contact is not continuously captured, referee signal or original call is not visible and not provided by the user, or the play may be a replay duplicate.",
    "If can_reason_about_call is false, missing_evidence must enumerate exactly what is missing in concrete visual terms (e.g., 'no frame shows the defender's feet at the moment of contact').",
    "limitations should list visual constraints that affect interpretation even when readiness is otherwise acceptable (e.g., 'partial occlusion of the ball-handler's lower body in frame_003').",
    "",
    "# FIELD-BY-FIELD CONTRACT (READ CAREFULLY)",
    "video_summary RULES:",
    "- Write video_summary as a single continuous neutral paragraph of plain descriptive prose, multiple sentences, in temporal order, as if narrating continuous footage.",
    "- DO NOT mention frame_id, frame numbers, the word 'frame', timestamps, or any reference to the selection, scoring, or recommendation process.",
    "- DO NOT mention Agent 1, Agent 2, the Verdict Agent, replay-readiness, can_reason_about_call, evidence quality, or any pipeline/metadata field.",
    "- DO NOT include rules reasoning, verdicts, fairness judgments, or call-type labels.",
    "- video_summary must remain self-contained and reusable as an embedding/search query. Avoid pipeline jargon. Describe only what is visibly happening.",
    "- Cover setting, players, ball state, contact if any, and how the action progresses, but as flowing prose only.",
    "Frame-bound fields (key_events, possible_critical_moments, recommended_frames_for_verdict_agent) are the ONLY fields that may cite frame_id and timestamps.",
    "- Each entry in those arrays MUST include the frame_id and timestamp it refers to.",
    "- Each key_events[].description must be a complete declarative sentence grounded in the cited frame.",
    "- Each possible_critical_moments[].description must be a complete sentence describing what is visible in the cited frame.",
    "- Recommend frames based only on visual clarity, sequence relevance, and whether they show before/during/after positions of the key moment.",
    "Quality/readiness fields (visual_frame_quality, verdict_readiness, can_reason_about_call, possible_replay_duplicate, sequence_interpretation, sequence_interpretation_reason, missing_evidence, limitations) live as their own structured values. Never narrate them inside video_summary.",
    "officiating_issue_summary may reference what a downstream reviewer should examine, but must remain neutral and visually grounded; it is NOT a verdict.",
    "",
    "# EXAMPLES OF GOOD VS FORBIDDEN video_summary",
    "GOOD: 'A white-jersey ball-handler drives along the right baseline; a dark-jersey defender slides into their path with both feet set, and the two make chest-to-chest contact, after which the ball-handler falls to the floor.'",
    "FORBIDDEN: 'In frame_003 at 0:02.50 the defender is set; this is recommended for the Verdict Agent.' (mentions frame_id, timestamp, and pipeline jargon)",
    "FORBIDDEN: 'This appears to be a charging foul because the defender was set first.' (rules reasoning / verdict)",
    "",
    "# OUTPUT REQUIREMENTS",
    "Return valid JSON only, with no prose outside the JSON, no markdown fences, and exactly this schema:",
    "{",
    '  "agent": "visual_analyst",',
    '  "video_summary": "string",',
    '  "key_events": [{"timestamp": "M:SS.ss", "frame_id": "frame_00x", "description": "string"}],',
    '  "sequence_interpretation": "single_play|multiple_plays|possible_replay|unclear",',
    '  "possible_replay_duplicate": false,',
    '  "sequence_interpretation_reason": "string",',
    '  "possible_critical_moments": [',
    '    {"timestamp": "M:SS.ss", "frame_id": "frame_00x", "description": "string", "why_relevant": "possible contact|defender position|ball release|referee signal|boundary|other"}',
    "  ],",
    '  "visible_call_type": "blocking/charging|traveling|goaltending|shooting foul|personal foul|out of bounds|no-call|unclear",',
    '  "visible_call_type_reason": "string",',
    '  "possible_call_types": ["blocking/charging", "personal foul"],',
    '  "officiating_issue_summary": "string",',
    '  "visual_frame_quality": "Good|Limited|Poor",',
    '  "verdict_readiness": "Ready|Limited|Not ready",',
    '  "can_reason_about_call": true,',
    '  "missing_evidence": ["string"],',
    '  "limitations": ["string"],',
    '  "recommended_frames_for_verdict_agent": [{"frame_id":"frame_00x","timestamp":"M:SS.ss","reason":"string"}]',
    "}",
]

_CRITICAL_MOMENT_WHY_MAP = {
    "high motion": "possible contact",
    "pre-contact context": "defender position",
    "post-contact context": "possible contact",
    "selected for temporal coverage": "other",
    "possible contact context": "defender position",
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
        "video_summary": "",
        "key_events": [],
        "possible_critical_moments": [],
        "sequence_interpretation": "unclear",
        "possible_replay_duplicate": False,
        "sequence_interpretation_reason": "",
        "visible_call_type": "unclear",
        "visible_call_type_reason": "",
        "possible_call_types": [],
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
        "error": None,
        "error_type": None,
        "model_used": model_name,
        "debug": {
            "model_used": model_name,
            "candidates_scanned_count": 0,
            "frames_extracted_count": 0,
            "frames_sent_to_gemini_count": 0,
            "selected_evidence_count": 0,
            "gemini_api_call_succeeded": False,
            "response_parsing_succeeded": False,
            "error_type": None,
            "error_message": None,
            "raw_response_excerpt": "",
            "scoring_top_candidates": [],
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
    try:
        scan_indices = _candidate_indices(total_frames=total_frames, fps=fps, duration=duration)
        result["debug"]["candidates_scanned_count"] = len(scan_indices)
        candidates = _collect_candidates(capture=capture, indices=scan_indices, fps=fps)
        if not candidates:
            raise RuntimeError("No candidates after scan.")
        selected = _select_evidence_candidates(candidates, target_min=8, target_max=12)
        if not selected:
            raise RuntimeError("Scoring produced no selected candidates.")
        result["debug"]["scoring_top_candidates"] = [
            {
                "timestamp": c["timestamp"],
                "reason": c.get("selection_reason", ""),
                "score": round(c.get("score", 0.0), 3),
            }
            for c in sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:5]
        ]
        return selected, "scored"
    except Exception:
        fallback_indices = _uniform_fallback_indices(total_frames, desired=8)
        fallback_candidates = _collect_candidates(capture=capture, indices=fallback_indices, fps=fps)
        for item in fallback_candidates:
            item["selection_reason"] = "selected for temporal coverage"
        result["debug"]["candidates_scanned_count"] = len(fallback_indices)
        return fallback_candidates, "uniform_fallback"


def _persist_evidence_frames(selected_candidates: list, frames_dir: Path, cv2_mod) -> list[dict]:
    evidence_frames = []
    for i, candidate in enumerate(selected_candidates, start=1):
        frame_name = f"{uuid4().hex}_{i}.jpg"
        frame_path = frames_dir / frame_name
        wrote = cv2_mod.imwrite(str(frame_path), candidate["frame"], [int(cv2_mod.IMWRITE_JPEG_QUALITY), 82])
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
            }
        )
    return evidence_frames


def _build_selection_summary(selection_mode: str, scanned_count: int, num_evidence: int) -> str:
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
    result["sequence_interpretation_reason"] = "No model output was available to distinguish a single play from a replay or separate play."


def _build_visual_prompt(evidence_frames: list[dict], original_call: str | None) -> str:
    lines = list(_VISUAL_PROMPT_INSTRUCTIONS)
    if original_call:
        lines.append(f'User-provided original call: "{original_call}".')
    lines.append("Selected evidence frames in temporal order:")
    for frame in evidence_frames:
        lines.append(
            f'- {frame["frame_id"]} at {frame["timestamp"]} -> {frame.get("selection_reason", "context")}'
        )
    return "\n".join(lines)


def _call_gemini(api_key: str, model_name: str, prompt: str, evidence_frames: list[dict], result: dict) -> str:
    """Upload frames, call Gemini, mutate debug fields, return raw response text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    uploaded_parts = [client.files.upload(file=frame["local_path"]) for frame in evidence_frames]
    result["debug"]["frames_sent_to_gemini_count"] = len(uploaded_parts)

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, *uploaded_parts],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result["debug"]["gemini_api_call_succeeded"] = True
    raw_text = (response.text or "").strip()
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
    result["sequence_interpretation_reason"] = "No model output was available to assess replay or sequence continuity."


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

    result["agent"] = str(payload.get("agent", "visual_analyst")).strip().lower() or "visual_analyst"
    result["video_summary"] = str(payload.get("video_summary", "")).strip()[:900]
    result["key_events"] = normalized_events
    result["possible_critical_moments"] = normalized_critical or result["possible_critical_moments"]
    result["sequence_interpretation"] = _normalize_sequence_interpretation(
        payload.get("sequence_interpretation", "")
    )
    result["possible_replay_duplicate"] = bool(payload.get("possible_replay_duplicate", False))
    result["sequence_interpretation_reason"] = str(
        payload.get("sequence_interpretation_reason", "")
    ).strip()[:500]
    result["visible_call_type"] = _normalize_call_type(payload.get("visible_call_type", ""))
    result["visible_call_type_reason"] = str(payload.get("visible_call_type_reason", "")).strip()[:500]
    result["possible_call_types"] = _normalize_possible_call_types(payload.get("possible_call_types") or [])
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
        and not result["possible_replay_duplicate"]
        and result["sequence_interpretation"] != "possible_replay"
    )
    result["missing_evidence"] = normalized_missing
    if requested_can_reason and not result["can_reason_about_call"]:
        if result["possible_replay_duplicate"] or result["sequence_interpretation"] == "possible_replay":
            result["missing_evidence"].append(
                "The sequence may be a replay duplicate, so later verdict reasoning should not treat it as a continuous single incident."
            )
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


# ----------------------------------------------------------------------------
# Agent 2: Verdict Agent
# ----------------------------------------------------------------------------

_VERDICT_PROMPT_INSTRUCTIONS = [
    "# ROLE",
    "You are Agent 2: Verdict Agent in a multi-agent officiating-review pipeline.",
    "You receive (a) a neutral visual description from Agent 1 (the Visual Analyst) and (b) a small set of curated still frames Agent 1 recommended for verdict review.",
    "Your job is to output ONE verdict on whether the on-court call appears Fair, Bad, or Inconclusive, grounded only in what the visuals plus the description support.",
    "You do NOT receive any user-provided original call. Reach a verdict independently.",
    "",
    "# YOUR JOB",
    "Integrate the visual evidence with standard NBA officiating expectations, including but not limited to: blocking vs. charging, traveling, goaltending and basket interference, shooting fouls, personal fouls, out-of-bounds, three-second / defensive three-second, and restricted-area rules.",
    "Cite frames by frame_id when stating what you see. You may quote short phrases from the Agent 1 video_summary when they support a claim.",
    "Treat repeated frames from a different camera angle as the same incident, not separate plays.",
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
    "- Do not consider any user-provided original call (you are not given one).",
    "- Confidence reflects how strongly the visible evidence supports your verdict label, not how certain you feel in general. Use Low when evidence is thin, Medium when partial, High only when the frames clearly settle the question.",
    "",
    "# OUTPUT REQUIREMENTS",
    "- All descriptions must be complete sentences.",
    "- key_factors must cite the frame_id (and timestamp) the factor is grounded in.",
    "- reasoning must be a single short paragraph of 3 to 6 sentences.",
    "- key_factors must contain between 2 and 5 entries.",
    "- rule_basis: a brief, plain-language statement of which officiating consideration drove the verdict (e.g., 'Defender appears set with both feet outside the restricted area before contact, consistent with a charge'). Keep it neutral and specific to what is visible.",
    "- limitations must list anything that constrained the verdict (occluded body parts, missing pre/post-contact frames, ambiguous body part of contact, etc.).",
    "Return valid JSON only, with no prose outside the JSON, no markdown fences, and exactly this schema:",
    "{",
    '  "agent": "verdict_agent",',
    '  "verdict": "Fair Call|Bad Call|Inconclusive",',
    '  "confidence": "Low|Medium|High",',
    '  "reasoning": "string",',
    '  "key_factors": [{"frame_id": "frame_00x", "timestamp": "M:SS.ss", "factor": "string"}],',
    '  "rule_basis": "string",',
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


def _new_verdict_result(model_name: str) -> dict:
    return {
        "agent": "verdict_agent",
        "success": False,
        "status": "Skipped",
        "skipped_reason": "",
        "verdict": "Inconclusive",
        "confidence": "Low",
        "reasoning": "",
        "key_factors": [],
        "rule_basis": "",
        "matched_rules": [],
        "limitations": [],
        "model_used": model_name,
        "error": None,
        "error_type": None,
        "debug": {
            "model_used": model_name,
            "frames_sent_to_gemini_count": 0,
            "gemini_api_call_succeeded": False,
            "response_parsing_succeeded": False,
            "raw_response_excerpt": "",
            "error_type": None,
            "error_message": None,
        },
    }


def _pick_verdict_frames(visual_result: dict) -> list[dict]:
    evidence_frames = visual_result.get("evidence_frames") or []
    recommended = visual_result.get("recommended_frames_for_verdict_agent") or []
    by_id = {f["frame_id"]: f for f in evidence_frames if isinstance(f, dict) and f.get("frame_id")}

    selected: list[dict] = []
    seen: set[str] = set()
    for rec in recommended:
        if not isinstance(rec, dict):
            continue
        frame_id = str(rec.get("frame_id", "")).strip()
        if frame_id in seen or frame_id not in by_id:
            continue
        base = by_id[frame_id]
        if not base.get("local_path"):
            continue
        seen.add(frame_id)
        selected.append(
            {
                "frame_id": frame_id,
                "timestamp": base.get("timestamp", rec.get("timestamp", "")),
                "local_path": base["local_path"],
                "reason": str(rec.get("reason", "")).strip()[:220]
                or base.get("selection_reason", "selected for temporal coverage"),
            }
        )

    if selected:
        return selected

    fallback = []
    for base in evidence_frames[:5]:
        if not isinstance(base, dict) or not base.get("local_path"):
            continue
        fallback.append(
            {
                "frame_id": base.get("frame_id", ""),
                "timestamp": base.get("timestamp", ""),
                "local_path": base["local_path"],
                "reason": base.get("selection_reason", "selected for temporal coverage"),
            }
        )
    return fallback


def _build_verdict_prompt(
    video_summary: str,
    frames_for_verdict: list[dict],
    advisory: dict | None = None,
    matched_rules: list[dict] | None = None,
) -> str:
    lines = list(_VERDICT_PROMPT_INSTRUCTIONS)
    lines.append("")
    lines.append("Visual Analyst summary (verbatim):")
    lines.append('"""')
    lines.append(video_summary.strip())
    lines.append('"""')
    lines.append("")
    lines.append("Curated frames for verdict review (in temporal order):")
    for frame in frames_for_verdict:
        lines.append(
            f'- {frame["frame_id"]} at {frame["timestamp"]} -> {frame.get("reason", "context")}'
        )
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
        if "possible_replay_duplicate" in advisory:
            lines.append(f"- possible_replay_duplicate: {bool(advisory.get('possible_replay_duplicate'))}")
        sequence = advisory.get("sequence_interpretation")
        if sequence:
            lines.append(f"- sequence_interpretation: {sequence}")
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
    from google.genai import types

    client = genai.Client(api_key=api_key)
    uploaded_parts = [client.files.upload(file=frame["local_path"]) for frame in frames_for_verdict]
    result["debug"]["frames_sent_to_gemini_count"] = len(uploaded_parts)

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, *uploaded_parts],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result["debug"]["gemini_api_call_succeeded"] = True
    raw_text = (response.text or "").strip()
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


def _apply_verdict_payload(result: dict, payload: dict, frame_id_map: dict) -> None:
    result["agent"] = str(payload.get("agent", "verdict_agent")).strip().lower() or "verdict_agent"
    result["verdict"] = _normalize_verdict_label(payload.get("verdict"))
    result["confidence"] = _normalize_verdict_confidence(payload.get("confidence"))
    result["reasoning"] = str(payload.get("reasoning", "")).strip()[:1200]
    result["key_factors"] = _normalize_verdict_factors(payload.get("key_factors"), frame_id_map)
    result["rule_basis"] = str(payload.get("rule_basis", "")).strip()[:500]
    raw_limitations = payload.get("limitations") or []
    result["limitations"] = [
        str(item).strip()[:220] for item in raw_limitations[:6] if str(item).strip()
    ]
    result["status"] = "Analyzed"
    result["success"] = True
    result["error"] = None
    result["error_type"] = None


def run_verdict_agent(visual_result: dict) -> dict:
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    result = _new_verdict_result(model_name)

    if visual_result.get("status") != "Analyzed":
        _apply_verdict_skipped(result, "Agent 1 did not produce an Analyzed result.")
        return result
    video_summary = (visual_result.get("video_summary") or "").strip()
    if not video_summary:
        _apply_verdict_skipped(result, "No video_summary available.")
        return result

    matched_rules = _retrieve_matching_rules(video_summary)
    result["matched_rules"] = matched_rules

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
        "possible_replay_duplicate": visual_result.get("possible_replay_duplicate"),
        "sequence_interpretation": visual_result.get("sequence_interpretation"),
        "missing_evidence": visual_result.get("missing_evidence") or [],
    }
    prompt = _build_verdict_prompt(video_summary, frames, advisory=advisory, matched_rules=matched_rules)
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
        _apply_verdict_payload(result, payload, frame_id_map)
    except json.JSONDecodeError as exc:
        _apply_verdict_json_decode_failure(result, raw_text, exc)
    except Exception as exc:
        _apply_verdict_api_failure(result, exc)

    return result


def analyze_video_with_gemini(video_path: str, original_call: str | None = None) -> dict:
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    result = _new_visual_result(model_name)

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
        selection_mode, result["debug"]["candidates_scanned_count"], len(evidence_frames)
    )
    result["possible_critical_moments"] = _initial_critical_moments(evidence_frames)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _apply_missing_api_key(result)
        _clean_for_session(result)
        return result

    _run_gemini_analysis(api_key, model_name, evidence_frames, original_call, result)
    result["verdict"] = run_verdict_agent(result)
    _clean_for_session(result)
    return result
