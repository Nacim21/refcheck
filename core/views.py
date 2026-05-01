from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect, render
from django.utils import timezone

from .services import analyze_video_with_gemini


def home(request):
    return render(request, "core/home.html")


MAX_VIDEO_SIZE_BYTES = 50 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
UPLOAD_RESULT_SESSION_KEY = "upload_result"


def _format_file_size(size_in_bytes):
    if size_in_bytes >= 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    if size_in_bytes >= 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    return f"{size_in_bytes} bytes"


def _status_message(analysis: dict) -> str:
    status = analysis.get("status")
    error_type = analysis.get("error_type")

    if status == "Analyzed":
        return "Video was analyzed from representative frames."
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


def analyze(request):
    context = {
        "form_data": {
            "original_call": "",
            "sport": "Basketball",
        },
        "errors": [],
    }

    if request.method == "POST":
        original_call = request.POST.get("original_call", "").strip()
        sport = request.POST.get("sport", "Basketball").strip() or "Basketball"
        video_file = request.FILES.get("video")

        context["form_data"]["original_call"] = original_call
        context["form_data"]["sport"] = sport

        errors = []

        if not video_file:
            errors.append("Please upload a video file.")
        else:
            extension = Path(video_file.name).suffix.lower()
            if extension not in ALLOWED_VIDEO_EXTENSIONS:
                errors.append("Unsupported file type. Please upload an MP4, MOV, or WEBM video.")
            if video_file.size > MAX_VIDEO_SIZE_BYTES:
                errors.append("File is too large. Maximum allowed size is 50 MB.")

        if errors:
            context["errors"] = errors
            return render(request, "core/analyze.html", context)

        storage = FileSystemStorage(
            location=settings.MEDIA_ROOT / "uploads",
            base_url=f"{settings.MEDIA_URL}uploads/",
        )
        unique_filename = f"{uuid4().hex}{Path(video_file.name).suffix.lower()}"
        stored_name = storage.save(unique_filename, video_file)

        stored_path = Path(storage.path(stored_name))
        started_at = timezone.now()
        ai_result = analyze_video_with_gemini(
            video_path=str(stored_path),
            original_call=original_call or None,
        )
        finished_at = timezone.now()

        request.session[UPLOAD_RESULT_SESSION_KEY] = {
            "video_url": storage.url(stored_name),
            "original_call": original_call or "Not provided",
            "sport": "Basketball",
            "filename": stored_name,
            "file_size": _format_file_size(video_file.size),
            "analysis": ai_result,
            "analysis_status": ai_result.get("status", "Failed"),
            "analysis_status_label": ai_result.get("status", "Failed"),
            "analysis_status_message": _status_message(ai_result),
            "analysis_started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return redirect("result")

    return render(request, "core/analyze.html", context)


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
