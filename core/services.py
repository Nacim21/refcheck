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
        "blocking",
        "charging",
        "traveling",
        "goaltending",
        "personal foul",
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
    # Cap candidates to keep payload and CPU bounded.
    max_candidates = 30
    if len(indices) > max_candidates:
        sampled = []
        for i in range(max_candidates):
            pos = int(round(i * (len(indices) - 1) / (max_candidates - 1)))
            sampled.append(indices[pos])
        indices = sampled
    # Avoid over-weighting exact 0.00 as primary moment; keep it only if needed.
    indices = sorted(set(indices))
    return indices


def _collect_candidates(capture, indices: list[int], fps: float):
    import cv2
    import numpy as np

    candidates = []
    prev_small_gray = None
    prev_hist = None

    for i, frame_idx in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        # Preserve quality for previews and Gemini, but cap width.
        max_width = 1024
        original_h, original_w = frame.shape[:2]
        if original_w > max_width:
            scale = max_width / float(original_w)
            resized_h = int(original_h * scale)
            frame = cv2.resize(frame, (max_width, resized_h), interpolation=cv2.INTER_AREA)

        # Lightweight analysis frame.
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        motion = 0.0
        if prev_small_gray is not None:
            diff = cv2.absdiff(gray, prev_small_gray)
            motion = float(diff.mean() / 255.0)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # Rough hardwood floor color heuristic.
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

    # Critical moments by motion.
    ranked_motion = sorted(candidates, key=lambda c: c["motion_norm"], reverse=True)
    top_motion = ranked_motion[:3]

    selected_ids = set()
    selected = []

    # Include pre/during/post around high motion candidates.
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

    # Fill remaining slots by score while ensuring diversity and reducing duplicates.
    ranked_score = sorted(candidates, key=lambda c: c["score"], reverse=True)
    for item in ranked_score:
        if len(selected) >= target_max:
            break
        if item["candidate_id"] in selected_ids:
            continue

        # Diversity gate: avoid frames too close in candidate order.
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
        # Backfill from temporal coverage.
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


def analyze_video_with_gemini(video_path: str, original_call: str | None = None) -> dict:
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    result = {
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
        "possible_critical_moments": [],
        "video_summary": "",
        "key_events": [],
        "visible_call_type": "unclear",
        "evidence_quality": "Poor",
        "can_reason_about_call": False,
        "missing_evidence": [],
        "officiating_issue_summary": "",
        "referee_signal_interpretation": "",
        "limitations": [],
        "confidence_in_visual_description": "Low",
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

    if not video_path:
        result["error_type"] = "video_not_found"
        result["error"] = "Uploaded video path is missing."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        return result

    video_file_path = Path(video_path)
    if not video_file_path.exists():
        result["error_type"] = "video_not_found"
        result["error"] = "Uploaded video file was not found."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        return result

    try:
        import cv2
    except Exception as exc:
        result["error_type"] = "frame_extraction_failure"
        result["error"] = "Video processing dependency is unavailable."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = str(exc)[:300]
        return result

    capture = cv2.VideoCapture(str(video_file_path))
    if not capture.isOpened():
        result["error_type"] = "frame_extraction_failure"
        result["error"] = "Uploaded video could not be opened."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        return result

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    result["metadata"] = {
        "duration_seconds": round(duration, 2) if duration > 0 else None,
        "fps": round(fps, 2) if fps > 0 else None,
        "total_frames": total_frames if total_frames > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
    }

    if total_frames <= 0:
        capture.release()
        result["error_type"] = "frame_extraction_failure"
        result["error"] = "No readable frames were found in the uploaded video."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        return result

    frames_dir = settings.MEDIA_ROOT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Candidate scan + scoring; fallback to uniform if this block fails.
    selected_candidates = []
    selection_mode = "scored"
    try:
        scan_indices = _candidate_indices(total_frames=total_frames, fps=fps, duration=duration)
        result["debug"]["candidates_scanned_count"] = len(scan_indices)
        candidates = _collect_candidates(capture=capture, indices=scan_indices, fps=fps)
        if not candidates:
            raise RuntimeError("No candidates after scan.")

        selected_candidates = _select_evidence_candidates(candidates, target_min=8, target_max=12)
        if not selected_candidates:
            raise RuntimeError("Scoring produced no selected candidates.")

        result["debug"]["scoring_top_candidates"] = [
            {
                "timestamp": candidate["timestamp"],
                "reason": candidate.get("selection_reason", ""),
                "score": round(candidate.get("score", 0.0), 3),
            }
            for candidate in sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:5]
        ]
    except Exception:
        selection_mode = "uniform_fallback"
        fallback_indices = _uniform_fallback_indices(total_frames, desired=8)
        fallback_candidates = _collect_candidates(capture=capture, indices=fallback_indices, fps=fps)
        for item in fallback_candidates:
            item["selection_reason"] = "selected for temporal coverage"
        selected_candidates = fallback_candidates
        result["debug"]["candidates_scanned_count"] = len(fallback_indices)
    finally:
        capture.release()

    if not selected_candidates:
        result["error_type"] = "frame_extraction_failure"
        result["error"] = "Could not extract representative frames from this video."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        return result

    evidence_frames = []
    for i, candidate in enumerate(selected_candidates):
        frame_name = f"{uuid4().hex}_{i}.jpg"
        frame_path = frames_dir / frame_name
        wrote = cv2.imwrite(
            str(frame_path),
            candidate["frame"],
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if not wrote:
            continue
        evidence_frames.append(
            {
                "timestamp": candidate["timestamp"],
                "frame_url": f"{settings.MEDIA_URL}frames/{frame_name}",
                "frame_path": f"frames/{frame_name}",
                "local_path": str(frame_path),
                "selection_reason": candidate.get("selection_reason", "selected for temporal coverage"),
            }
        )

    result["evidence_frames"] = evidence_frames
    result["status"] = "Partial"
    result["debug"]["frames_extracted_count"] = len(evidence_frames)
    result["debug"]["selected_evidence_count"] = len(evidence_frames)

    if selection_mode == "scored":
        result["evidence_selection_summary"] = (
            f"Scanned {result['debug']['candidates_scanned_count']} candidate frames and selected "
            f"{len(evidence_frames)} evidence frames using motion, sharpness, duplicate filtering, "
            "and temporal context around likely action moments."
        )
    else:
        result["evidence_selection_summary"] = (
            f"Scoring fallback triggered. Selected {len(evidence_frames)} uniformly distributed evidence frames."
        )

    critical_moments = []
    for frame in evidence_frames[:6]:
        reason = frame.get("selection_reason", "possible contact context")
        why_map = {
            "high motion": "possible contact",
            "pre-contact context": "player positioning",
            "post-contact context": "possible contact",
            "selected for temporal coverage": "unclear",
            "possible contact context": "player positioning",
        }
        critical_moments.append(
            {
                "timestamp": frame["timestamp"],
                "description": f"Frame selected as {reason}.",
                "why_relevant": why_map.get(reason, "unclear"),
            }
        )
    result["possible_critical_moments"] = critical_moments

    if not evidence_frames:
        result["error_type"] = "frame_extraction_failure"
        result["error"] = "Could not save selected evidence frames."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        return result

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        result["error_type"] = "missing_api_key"
        result["error"] = "GEMINI_API_KEY is not configured. Frame extraction worked, but AI analysis is unavailable."
        result["limitations"] = ["AI description unavailable because Gemini API key is missing."]
        result["missing_evidence"] = [
            "No model-generated officiating interpretation was produced.",
            "Before/contact/after continuity cannot be assessed without AI output.",
        ]
        result["officiating_issue_summary"] = "Evidence is insufficient because no AI interpretation was returned."
        result["referee_signal_interpretation"] = "No reliable referee signal interpretation is available."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = result["error"]
        _clean_for_session(result)
        return result

    prompt_lines = [
        "You are analyzing selected evidence frames from a basketball video.",
        "These frames were selected around likely action/contact moments and may not cover the full continuous clip.",
        "Use the timestamps to infer before/during/after sequence when possible.",
        "Focus on officiating-relevant evidence: contact, defender position, offensive player movement, ball location, referee signals, and sequence around possible call.",
        "Distinguish visible game events from officiating-relevant evidence.",
        "A referee raising an arm must not automatically be interpreted as a foul.",
        "Scoreboard changes must not be treated as proof of a correct or incorrect call.",
        "If before/contact/after continuity is missing, mark evidence_quality as Limited or Poor.",
        "If you cannot reason about the call from selected frames, set can_reason_about_call to false.",
        "Explain exactly what evidence is missing.",
        "Explicitly state if selected frames are insufficient.",
        "Do not give a final Fair Call or Bad Call verdict.",
        "Do not cite rules.",
        "Return valid JSON only with this schema:",
        "{",
        '  "video_summary": "string",',
        '  "key_events": [{"timestamp": "M:SS.ss", "description": "string"}],',
        '  "visible_call_type": "blocking|charging|traveling|goaltending|personal foul|no-call|unclear",',
        '  "limitations": ["string"],',
        '  "confidence_in_visual_description": "Low|Medium|High",',
        '  "evidence_selection_summary": "string",',
        '  "evidence_quality": "Good|Limited|Poor",',
        '  "can_reason_about_call": true,',
        '  "missing_evidence": ["string"],',
        '  "officiating_issue_summary": "string",',
        '  "referee_signal_interpretation": "string",',
        '  "possible_critical_moments": [',
        '    {"timestamp": "M:SS.ss", "description": "string", "why_relevant": "possible contact|player positioning|referee signal|unclear"}',
        "  ]",
        "}",
    ]
    if original_call:
        prompt_lines.append(f'User-provided original call: "{original_call}".')
    prompt_lines.append("Evidence frames (timestamp -> reason):")
    for frame in evidence_frames:
        prompt_lines.append(f'- {frame["timestamp"]} -> {frame.get("selection_reason", "context")}')
    prompt = "\n".join(prompt_lines)

    raw_text = ""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        uploaded_parts = []
        for frame in evidence_frames:
            uploaded_parts.append(client.files.upload(file=frame["local_path"]))
        result["debug"]["frames_sent_to_gemini_count"] = len(uploaded_parts)

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, *uploaded_parts],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        result["debug"]["gemini_api_call_succeeded"] = True
        raw_text = (response.text or "").strip()
        result["debug"]["raw_response_excerpt"] = _safe_excerpt(raw_text)

        if not raw_text:
            result["error_type"] = "empty_gemini_response"
            result["error"] = "AI returned an empty response."
            result["limitations"] = ["The AI response was empty for this request."]
            result["missing_evidence"] = [
                "No sequence interpretation was returned by the model.",
                "Contact continuity could not be evaluated.",
            ]
            result["officiating_issue_summary"] = "No officiating evidence interpretation was returned."
            result["referee_signal_interpretation"] = "No referee signal interpretation available."
            result["debug"]["error_type"] = result["error_type"]
            result["debug"]["error_message"] = result["error"]
            _clean_for_session(result)
            return result

        json_candidate = _extract_json_object(raw_text)
        if not json_candidate:
            result["error_type"] = "json_parsing_failure"
            result["video_summary"] = _strip_code_fences(raw_text)[:900]
            result["limitations"] = [
                "Structured JSON parsing failed; showing raw AI summary text instead.",
                "Selected evidence frames may be insufficient for full temporal reconstruction.",
            ]
            result["missing_evidence"] = [
                "Structured evidence-sufficiency fields were not returned in JSON.",
                "Continuous before/contact/after sequence remains uncertain.",
            ]
            result["officiating_issue_summary"] = "Unstructured output; officiating evidence sufficiency is uncertain."
            result["referee_signal_interpretation"] = "Referee signal meaning is uncertain from unstructured output."
            result["error"] = "AI returned unstructured text."
            result["debug"]["error_type"] = result["error_type"]
            result["debug"]["error_message"] = "No JSON object found in Gemini output."
            _clean_for_session(result)
            return result

        payload = json.loads(json_candidate)
        result["debug"]["response_parsing_succeeded"] = True

        key_events = payload.get("key_events") or []
        limitations = payload.get("limitations") or []
        critical_payload = payload.get("possible_critical_moments") or []

        normalized_events = []
        for event in key_events[:6]:
            if not isinstance(event, dict):
                continue
            timestamp = str(event.get("timestamp", "")).strip()
            description = str(event.get("description", "")).strip()
            if timestamp and description:
                normalized_events.append(
                    {
                        "timestamp": timestamp[:20],
                        "description": description[:220],
                    }
                )

        normalized_limitations = [
            str(item).strip()[:220]
            for item in limitations[:6]
            if str(item).strip()
        ]
        missing_evidence = payload.get("missing_evidence") or []
        normalized_missing_evidence = [
            str(item).strip()[:220]
            for item in missing_evidence[:6]
            if str(item).strip()
        ]

        normalized_critical = []
        for item in critical_payload[:8]:
            if not isinstance(item, dict):
                continue
            timestamp = str(item.get("timestamp", "")).strip()
            description = str(item.get("description", "")).strip()
            why = str(item.get("why_relevant", "unclear")).strip().lower()
            if not timestamp or not description:
                continue
            if why not in {"possible contact", "player positioning", "referee signal", "unclear"}:
                why = "unclear"
            normalized_critical.append(
                {
                    "timestamp": timestamp[:20],
                    "description": description[:220],
                    "why_relevant": why,
                }
            )

        result["video_summary"] = str(payload.get("video_summary", "")).strip()[:900]
        result["key_events"] = normalized_events
        result["visible_call_type"] = _normalize_call_type(payload.get("visible_call_type", ""))
        result["evidence_quality"] = _normalize_evidence_quality(payload.get("evidence_quality", ""))
        result["can_reason_about_call"] = bool(payload.get("can_reason_about_call", False))
        result["missing_evidence"] = normalized_missing_evidence
        result["officiating_issue_summary"] = str(payload.get("officiating_issue_summary", "")).strip()[:500]
        result["referee_signal_interpretation"] = str(payload.get("referee_signal_interpretation", "")).strip()[:500]
        result["limitations"] = normalized_limitations
        result["confidence_in_visual_description"] = _normalize_confidence(
            payload.get("confidence_in_visual_description", "")
        )
        payload_selection_summary = str(payload.get("evidence_selection_summary", "")).strip()
        if payload_selection_summary:
            result["evidence_selection_summary"] = payload_selection_summary[:400]
        if normalized_critical:
            result["possible_critical_moments"] = normalized_critical

        if not result["video_summary"]:
            result["error_type"] = "empty_gemini_response"
            result["error"] = "AI response was received but summary text was empty."
            result["limitations"] = ["AI returned structured data without a usable summary."]
            result["missing_evidence"] = [
                "No usable sequence summary was returned.",
                "Before/contact/after continuity remains unresolved.",
            ]
            result["officiating_issue_summary"] = "Insufficient summary to assess officiating context."
            result["referee_signal_interpretation"] = "No reliable referee signal interpretation returned."
            result["debug"]["error_type"] = result["error_type"]
            result["debug"]["error_message"] = result["error"]
            _clean_for_session(result)
            return result

        result["success"] = True
        result["status"] = "Analyzed"
        result["error"] = None
        result["error_type"] = None
        _clean_for_session(result)
        return result
    except json.JSONDecodeError as exc:
        result["error_type"] = "json_parsing_failure"
        result["video_summary"] = _strip_code_fences(raw_text)[:900]
        result["limitations"] = [
            "Structured JSON parsing failed; showing raw AI summary text instead.",
            "Some expected analysis fields may be missing.",
        ]
        result["missing_evidence"] = [
            "Evidence sufficiency fields could not be parsed.",
            "Continuous contact sequence remains uncertain.",
        ]
        result["officiating_issue_summary"] = "JSON parsing failed; evidence sufficiency is uncertain."
        result["referee_signal_interpretation"] = "Referee signal interpretation unavailable due to parsing failure."
        result["error"] = "AI output format could not be parsed as JSON."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = str(exc)[:300]
        _clean_for_session(result)
        return result
    except Exception as exc:
        result["error_type"] = "gemini_api_failure"
        result["error"] = "AI analysis could not be completed for this upload."
        result["limitations"] = ["Gemini API request failed before structured analysis could be produced."]
        result["missing_evidence"] = [
            "Model response was not received successfully.",
            "Before/contact/after continuity could not be evaluated.",
        ]
        result["officiating_issue_summary"] = "Gemini request failed; officiating evidence assessment was not completed."
        result["referee_signal_interpretation"] = "No referee signal interpretation available."
        result["debug"]["error_type"] = result["error_type"]
        result["debug"]["error_message"] = str(exc)[:300]
        _clean_for_session(result)
        return result
