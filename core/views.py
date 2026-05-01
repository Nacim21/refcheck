import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .services import analyze_video_with_gemini


MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}
UPLOAD_RESULT_SESSION_KEY = "upload_result"
_GCS_CORS_CONFIGURED = False


def _format_file_size(size_in_bytes):
    try:
        size_in_bytes = int(size_in_bytes)
    except (TypeError, ValueError):
        return "Not available"
    if size_in_bytes >= 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    if size_in_bytes >= 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    return f"{size_in_bytes} bytes"


def _status_message(analysis: dict) -> str:
    status = analysis.get("status")
    error_type = analysis.get("error_type")

    if status == "Analyzed":
        base = "Video was analyzed from representative frames."
        verdict = analysis.get("verdict") or {}
        verdict_status = verdict.get("status")
        if verdict_status == "Analyzed":
            label = verdict.get("verdict", "Inconclusive")
            confidence = verdict.get("confidence", "Low")
            return f"Verdict: {label} ({confidence} confidence)."
        if verdict_status == "Skipped":
            reason = verdict.get("skipped_reason") or "preconditions not met."
            return f"{base} Verdict skipped: {reason}"
        if verdict_status == "Failed":
            return f"{base} Verdict unavailable."
        return base
    if error_type == "missing_api_key":
        return "Video processing completed, but AI description is unavailable because Gemini API key is missing."
    if error_type == "frame_extraction_failure":
        return "Video upload succeeded, but frame extraction failed for this file."
    if error_type == "json_parsing_failure":
        return "Video frames were analyzed, but structured fields could not be parsed completely."
    if error_type == "empty_gemini_response":
        return "Gemini returned an empty response for this upload."
    if error_type == "gemini_api_failure":
        return "Video upload and frame extraction succeeded, but Gemini request failed."
    return "Video upload succeeded, but analysis could not be completed for this upload."


def _build_context(active_tab: str = "home") -> dict:
    return {
        "form_data": {
            "original_call": "",
            "sport": "Basketball",
        },
        "errors": [],
        "upload_result": None,
        "debug_enabled": settings.DEBUG,
        "active_tab": active_tab,
    }


def _json_payload(request) -> tuple[dict, str | None]:
    try:
        return json.loads(request.body.decode("utf-8") or "{}"), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, "Invalid JSON payload."


def _validate_video_metadata(filename: str, content_type: str) -> tuple[str | None, list[str]]:
    errors = []
    filename = (filename or "").strip()
    content_type = (content_type or "").strip().lower()
    extension = Path(filename).suffix.lower()

    if not filename:
        errors.append("Filename is required.")
    if content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        errors.append("Unsupported content type. Please upload MP4, MOV, or WEBM video.")
    if extension and extension not in ALLOWED_VIDEO_EXTENSIONS:
        errors.append("Unsupported file extension. Please upload an MP4, MOV, or WEBM video.")

    if errors:
        return None, errors
    return extension or ALLOWED_VIDEO_CONTENT_TYPES[content_type], []


def _gcs_bucket_name() -> str:
    return (
        os.getenv("GCP_STORAGE_BUCKET_NAME")
        or os.getenv("GCS_BUCKET_NAME")
        or ""
    ).strip()


def _gcp_credentials_info() -> dict | None:
    raw_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    if not raw_credentials:
        return None
    try:
        return json.loads(raw_credentials)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON.") from exc


def _get_gcs_bucket():
    bucket_name = _gcs_bucket_name()
    if not bucket_name:
        raise RuntimeError("GCP_STORAGE_BUCKET_NAME or GCS_BUCKET_NAME is not configured.")
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError("google-cloud-storage is not installed.") from exc
    credentials_info = _gcp_credentials_info()
    if credentials_info:
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return storage.Client(
            credentials=credentials,
            project=credentials_info.get("project_id"),
        ).bucket(bucket_name)
    return storage.Client().bucket(bucket_name)


def _origin_from_url(value: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(value or "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _allowed_cors_origins(request) -> list[str]:
    origins = {
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    }
    request_origin = request.headers.get("Origin", "")
    if request_origin:
        origins.add(request_origin.rstrip("/"))
    configured_origins = os.getenv("REFCHECK_GCS_CORS_ORIGINS", "")
    for origin in configured_origins.split(","):
        origin = origin.strip().rstrip("/")
        if origin:
            origins.add(origin)
    app_origin = _origin_from_url(os.getenv("NEXT_PUBLIC_APP_URL", ""))
    if app_origin:
        origins.add(app_origin)
    vercel_url = os.getenv("VERCEL_URL", "").strip()
    if vercel_url:
        origins.add(f"https://{vercel_url}".rstrip("/"))
    return sorted(origins)


def _ensure_gcs_cors(request) -> str:
    global _GCS_CORS_CONFIGURED
    if _GCS_CORS_CONFIGURED:
        return ""

    bucket = _get_gcs_bucket()
    desired_rule = {
        "origin": _allowed_cors_origins(request),
        "method": ["PUT", "GET", "HEAD", "OPTIONS"],
        "responseHeader": ["Content-Type", "x-goog-resumable"],
        "maxAgeSeconds": 3600,
    }
    existing_rules = bucket.cors or []
    for rule in existing_rules:
        if (
            set(rule.get("origin", [])) >= set(desired_rule["origin"])
            and set(rule.get("method", [])) >= {"PUT", "OPTIONS"}
            and "Content-Type" in set(rule.get("responseHeader", []))
        ):
            _GCS_CORS_CONFIGURED = True
            return ""

    bucket.cors = [*existing_rules, desired_rule]
    bucket.patch()
    _GCS_CORS_CONFIGURED = True
    return ""


def _generate_signed_upload_url(gcs_path: str, content_type: str) -> str:
    return _get_gcs_bucket().blob(gcs_path).generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        method="PUT",
        content_type=content_type,
    )


def _generate_signed_read_url(gcs_path: str) -> str:
    return _get_gcs_bucket().blob(gcs_path).generate_signed_url(
        version="v4",
        expiration=timedelta(hours=6),
        method="GET",
    )


def _validate_gcs_path(gcs_path: str) -> list[str]:
    gcs_path = (gcs_path or "").strip()
    if not gcs_path:
        return ["video_gcs_path is required."]
    if not gcs_path.startswith("uploads/") or ".." in gcs_path or gcs_path.endswith("/"):
        return ["Invalid GCS object path."]
    if Path(gcs_path).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return ["Unsupported GCS video extension."]
    return []


def _download_gcs_video_to_temp(gcs_path: str) -> str:
    suffix = Path(gcs_path).suffix.lower() or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(prefix="refcheck-gcs-", suffix=suffix, delete=False)
    temp_path = temp_file.name
    temp_file.close()
    _get_gcs_bucket().blob(gcs_path).download_to_filename(temp_path)
    return temp_path


@require_POST
def create_upload_url(request):
    payload, error = _json_payload(request)
    if error:
        return JsonResponse({"error": error}, status=400)

    filename = str(payload.get("filename", "")).strip()
    content_type = str(payload.get("content_type", "")).strip().lower()
    extension, errors = _validate_video_metadata(filename, content_type)
    if errors:
        return JsonResponse({"errors": errors}, status=400)

    gcs_path = f"uploads/{uuid4().hex}{extension}"
    cors_warning = ""
    try:
        try:
            cors_warning = _ensure_gcs_cors(request)
        except Exception as exc:
            cors_warning = f"GCS CORS could not be auto-configured: {exc}"
        upload_url = _generate_signed_upload_url(gcs_path, content_type)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    payload = {"upload_url": upload_url, "gcs_path": gcs_path}
    if cors_warning:
        payload["cors_warning"] = cors_warning
    return JsonResponse(payload)


def _store_local_upload(video_file) -> tuple[Path, str, str]:
    storage = FileSystemStorage(
        location=settings.MEDIA_ROOT / "uploads",
        base_url=f"{settings.MEDIA_URL}uploads/",
    )
    unique_filename = f"{uuid4().hex}{Path(video_file.name).suffix.lower()}"
    stored_name = storage.save(unique_filename, video_file)
    return Path(storage.path(stored_name)), stored_name, storage.url(stored_name)


def _build_upload_result(
    *,
    video_url: str,
    original_call: str,
    sport: str,
    filename: str,
    file_size: str,
    ai_result: dict,
    started_at,
    finished_at,
) -> dict:
    return {
        "video_url": video_url,
        "original_call": original_call or "Not provided",
        "sport": sport or "Basketball",
        "filename": filename,
        "file_size": file_size,
        "analysis": ai_result,
        "analysis_status": ai_result.get("status", "Failed"),
        "analysis_status_label": ai_result.get("status", "Failed"),
        "analysis_status_message": _status_message(ai_result),
        "analysis_started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _run_analysis_for_local_upload(video_file, original_call: str, sport: str) -> dict:
    stored_path, stored_name, video_url = _store_local_upload(video_file)
    started_at = timezone.now()
    ai_result = analyze_video_with_gemini(
        video_path=str(stored_path),
        original_call=original_call or None,
    )
    finished_at = timezone.now()
    return _build_upload_result(
        video_url=video_url,
        original_call=original_call,
        sport=sport,
        filename=stored_name,
        file_size=_format_file_size(video_file.size),
        ai_result=ai_result,
        started_at=started_at,
        finished_at=finished_at,
    )


def _run_analysis_for_gcs_upload(payload: dict, original_call: str, sport: str) -> tuple[dict | None, list[str]]:
    gcs_path = str(payload.get("video_gcs_path", "")).strip()
    errors = _validate_gcs_path(gcs_path)
    if errors:
        return None, errors

    temp_path = ""
    try:
        temp_path = _download_gcs_video_to_temp(gcs_path)
        started_at = timezone.now()
        ai_result = analyze_video_with_gemini(
            video_path=temp_path,
            original_call=original_call or None,
        )
        finished_at = timezone.now()
        try:
            video_url = _generate_signed_read_url(gcs_path)
        except Exception:
            video_url = ""
        return _build_upload_result(
            video_url=video_url,
            original_call=original_call,
            sport=sport,
            filename=gcs_path,
            file_size=_format_file_size(payload.get("file_size")),
            ai_result=ai_result,
            started_at=started_at,
            finished_at=finished_at,
        ), []
    except Exception as exc:
        return None, [f"Could not download video from GCS: {exc}"]
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _handle_analyze_post(request, context: dict, template_name: str):
    is_json_request = (request.content_type or "").split(";")[0] == "application/json"
    payload = {}
    if is_json_request:
        payload, error = _json_payload(request)
        if error:
            return JsonResponse({"errors": [error]}, status=400)
        original_call = str(payload.get("initial_call") or payload.get("original_call") or "").strip()
        sport = str(payload.get("sport") or "Basketball").strip() or "Basketball"
    else:
        original_call = request.POST.get("original_call", "").strip()
        sport = request.POST.get("sport", "Basketball").strip() or "Basketball"
        payload = {"video_gcs_path": request.POST.get("video_gcs_path", "").strip()}

    video_file = request.FILES.get("video")
    video_gcs_path = str(payload.get("video_gcs_path", "")).strip()

    context["form_data"]["original_call"] = original_call
    context["form_data"]["sport"] = sport
    context["active_tab"] = "analyze"

    errors = []
    upload_result = None

    if video_gcs_path:
        upload_result, errors = _run_analysis_for_gcs_upload(payload, original_call, sport)
    elif video_file:
        extension = Path(video_file.name).suffix.lower()
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            errors.append("Unsupported file type. Please upload an MP4, MOV, or WEBM video.")
        if video_file.size > MAX_VIDEO_SIZE_BYTES:
            errors.append("File is too large. Maximum allowed size is 100 MB.")
        if not errors:
            upload_result = _run_analysis_for_local_upload(video_file, original_call, sport)
    else:
        errors.append("Please upload a video file.")

    if errors:
        if is_json_request:
            return JsonResponse({"errors": errors}, status=400)
        context["errors"] = errors
        return render(request, template_name, context)

    request.session[UPLOAD_RESULT_SESSION_KEY] = upload_result
    if is_json_request:
        return JsonResponse({"redirect_url": reverse("result")})

    context["upload_result"] = upload_result
    context["active_tab"] = "live_feed"
    return render(request, template_name, context)


def home(request):
    context = _build_context(active_tab=request.GET.get("tab", "home"))
    if request.method == "POST":
        return _handle_analyze_post(request, context, "core/home.html")
    if context["active_tab"] in {"analyze", "live_feed"}:
        context["upload_result"] = request.session.get(UPLOAD_RESULT_SESSION_KEY)
    return render(request, "core/home.html", context)


def analyze(request):
    context = _build_context(active_tab="analyze")
    if request.method == "POST":
        return _handle_analyze_post(request, context, "core/home.html")
    context["upload_result"] = request.session.get(UPLOAD_RESULT_SESSION_KEY)
    return render(request, "core/home.html", context)


def result(request):
    upload_result = request.session.get(UPLOAD_RESULT_SESSION_KEY)
    return render(
        request,
        "core/result.html",
        {
            "upload_result": upload_result,
            "debug_enabled": settings.DEBUG,
        },
    )


def about(request):
    return render(request, "core/about.html")
