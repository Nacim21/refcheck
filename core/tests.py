import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse


class HomeFlowTests(TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="refcheck-tests-")
        self.media_root = Path(self._tmp_dir)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @override_settings(MEDIA_URL="/media/")
    def test_home_includes_live_feed_tab(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="tab-btn-live-feed"')
        self.assertContains(response, 'id="tab-content-live-feed"')

    @override_settings(MEDIA_URL="/media/")
    @patch("core.views.analyze_video_with_gemini")
    def test_upload_from_evidence_log_activates_live_feed(self, mocked_analyze):
        mocked_analyze.return_value = {
            "status": "Analyzed",
            "metadata": {"duration_seconds": 8.0, "fps": 30.0, "total_frames": 240},
            "video_summary": "Test summary.",
            "key_events": [],
            "evidence_frames": [],
        }
        file_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41"
        upload = SimpleUploadedFile("play.mp4", file_bytes, content_type="video/mp4")

        with override_settings(MEDIA_ROOT=self.media_root):
            response = self.client.post(
                reverse("home"),
                data={
                    "video": upload,
                    "original_call": "Goaltending",
                    "sport": "Basketball",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'const initialTabFromServer = "live_feed";')
        self.assertContains(response, "Live Feed Results")
        self.assertContains(response, "Analysis Result")
