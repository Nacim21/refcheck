from django.test import SimpleTestCase
from unittest.mock import patch

from . import services


class VisualSeverityTriageTests(SimpleTestCase):
    def test_primary_event_fallback_uses_highest_tier_then_latest_same_tier(self):
        events = [
            {
                "tier": "Tier C",
                "timestamp": "0:01.00",
                "frame_ids": ["frame_001"],
                "brief_description": "Players lightly jostle for position.",
            },
            {
                "tier": "Tier A",
                "timestamp": "0:03.00",
                "frame_ids": ["frame_003"],
                "brief_description": "An arm contacts an opponent near the face.",
            },
            {
                "tier": "Tier A",
                "timestamp": "0:04.00",
                "frame_ids": ["frame_004"],
                "brief_description": "A later hand contact is visible near the opponent's face.",
            },
        ]

        primary = services._primary_event_from_detected_events(events)

        self.assertEqual(primary["tier"], "Tier A")
        self.assertEqual(primary["timestamp"], "0:04.00")
        self.assertEqual(primary["frame_ids"], ["frame_004"])

    def test_payload_applies_detected_events_and_primary_event(self):
        result = services._new_visual_result("test-model")
        evidence_frames = [
            {"frame_id": "frame_001", "timestamp": "0:01.00"},
            {"frame_id": "frame_002", "timestamp": "0:04.00"},
        ]
        payload = {
            "agent": "visual_analyst",
            "all_events_detected": [
                {
                    "tier": "A",
                    "timestamp": "0:04.00",
                    "frame_ids": ["frame_002", "missing_frame"],
                    "brief_description": "Open-hand contact appears near the opponent's face.",
                }
            ],
            "primary_event": {
                "tier": "Tier A",
                "timestamp": "0:04.00",
                "frame_ids": ["frame_002"],
                "description": "Open-hand contact appears near the opponent's face.",
                "why_this_is_primary": "It is the highest-tier event visible in the selected frames.",
            },
            "video_summary": "Players jostle before open-hand contact appears near the face.",
            "key_events": [],
            "possible_critical_moments": [],
            "sequence_interpretation": "single_play",
            "visible_call_type": "unclear",
            "possible_call_types": ["personal foul"],
            "issue_cards": [],
            "visual_frame_quality": "Good",
            "verdict_readiness": "Ready",
            "can_reason_about_call": True,
            "confidence_in_visual_description": "Medium",
        }

        services._apply_payload_to_result(result, payload, evidence_frames)

        self.assertEqual(result["all_events_detected"][0]["tier"], "Tier A")
        self.assertEqual(result["all_events_detected"][0]["frame_ids"], ["frame_002"])
        self.assertEqual(result["primary_event"]["tier"], "Tier A")
        self.assertEqual(result["primary_event"]["frame_ids"], ["frame_002"])

    def test_visual_prompt_contains_severity_triage_contract(self):
        prompt_text = "\n".join(services._VISUAL_PROMPT_INSTRUCTIONS)

        self.assertIn("Pass A - Triage", prompt_text)
        self.assertIn("Tier A", prompt_text)
        self.assertIn("The primary event is the highest-tier event in the clip", prompt_text)
        self.assertIn("all_events_detected", prompt_text)
        self.assertIn("primary_event", prompt_text)

    def test_goaltending_selected_call_stays_primary_and_demotes_contact(self):
        result = services._new_visual_result("test-model")
        result["original_call"] = "Goaltending"
        evidence_frames = [
            {"frame_id": "frame_001", "timestamp": "0:01.00"},
            {"frame_id": "frame_002", "timestamp": "0:02.00"},
            {"frame_id": "frame_003", "timestamp": "0:03.00"},
        ]
        payload = {
            "agent": "visual_analyst",
            "video_summary": "White offense attacks rim while dark defender contests near the basket.",
            "sequence_interpretation": "possible_replay",
            "possible_replay_duplicate": True,
            "sequence_interpretation_reason": "Players look similar across adjacent frames.",
            "issue_cards": [
                {
                    "issue_id": "issue_001",
                    "issue_type": "personal_foul",
                    "description": "Possible body contact on airborne defender.",
                    "frame_ids": ["frame_002"],
                    "confidence": "Medium",
                    "needs_rule_retrieval": True,
                    "rag_query": "basketball personal foul airborne contact",
                }
            ],
            "can_reason_about_call": False,
            "verdict_readiness": "Limited",
        }

        services._apply_payload_to_result(result, payload, evidence_frames)

        self.assertEqual(result["primary_issue"], "goaltending")
        self.assertFalse(result["possible_replay_duplicate"])
        self.assertIn(result["sequence_interpretation"], {"single_play", "uncertain_continuity"})
        self.assertTrue(result["issue_cards"])
        self.assertEqual(result["issue_cards"][0]["issue_type"], "goaltending")
        self.assertTrue(result["secondary_issues"])
        self.assertIn("secondary", result["secondary_issues"][0]["reason"].lower())
        self.assertTrue(any("downward flight" in item.lower() for item in result["missing_evidence"]))


class AdaptiveFrameSelectionTests(SimpleTestCase):
    def test_adaptive_selection_concentrates_budget_on_primary_peak(self):
        fps = 10.0
        candidates = []
        for idx, motion in enumerate([0.05, 0.9, 0.08, 0.2, 0.1, 0.55, 0.1, 0.3]):
            frame_idx = idx * 10
            candidates.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": services._format_timestamp(frame_idx / fps),
                    "motion": motion,
                }
            )

        indices, debug = services._adaptive_indices_around_motion_peaks(
            candidates,
            total_frames=100,
            fps=fps,
            total_budget=12,
        )

        self.assertTrue(indices)
        self.assertEqual(debug["allocations"][0]["role"], "primary")
        self.assertEqual(debug["allocations"][0]["budget"], 9)
        self.assertTrue(all(item["motion"] >= 0.45 for item in debug["peak_ranking"][:2]))
        self.assertTrue(any(item["role"] == "secondary" for item in debug["allocations"]))

    def test_adaptive_selection_uses_full_budget_when_secondaries_are_weak(self):
        fps = 10.0
        candidates = []
        for idx, motion in enumerate([0.05, 1.0, 0.04, 0.2, 0.03, 0.3, 0.02]):
            frame_idx = idx * 10
            candidates.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": services._format_timestamp(frame_idx / fps),
                    "motion": motion,
                }
            )

        _, debug = services._adaptive_indices_around_motion_peaks(
            candidates,
            total_frames=100,
            fps=fps,
            total_budget=12,
        )

        self.assertEqual(len(debug["allocations"]), 1)
        self.assertEqual(debug["allocations"][0]["role"], "primary")
        self.assertEqual(debug["allocations"][0]["budget"], 12)

    def test_refcheck_frame_config_env_resolution(self):
        env = {
            "REFCHECK_SCAN_FPS": "4",
            "REFCHECK_DENSE_FPS": "18",
            "REFCHECK_DENSE_WINDOW_SECONDS": "1.5",
            "REFCHECK_MAX_EVIDENCE_FRAMES": "20",
            "REFCHECK_MIN_EVIDENCE_FRAMES": "12",
            "REFCHECK_MAX_VERDICT_FRAMES": "7",
            "REFCHECK_MAX_GEMINI_IMAGES": "24",
            "REFCHECK_MAX_CANDIDATE_FRAMES": "90",
            "REFCHECK_REPLAY_CONTEXT_ENABLED": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            self.assertEqual(services._refcheck_scan_fps(), 4.0)
            self.assertEqual(services._refcheck_dense_fps(), 18.0)
            self.assertEqual(services._refcheck_dense_window_seconds(), 1.5)
            self.assertEqual(services._refcheck_max_evidence_frames(), 20)
            self.assertEqual(services._refcheck_min_evidence_frames(), 12)
            self.assertEqual(services._refcheck_max_verdict_frames(), 7)
            self.assertEqual(services._refcheck_max_gemini_images(), 24)
            self.assertEqual(services._refcheck_max_candidate_frames(), 90)
            self.assertTrue(services._refcheck_replay_context_enabled())

    def test_verdict_frame_picker_caps_and_prioritizes_issue_frames(self):
        evidence_frames = [
            {
                "frame_id": f"frame_{index:03d}",
                "timestamp": f"0:0{index}.00",
                "local_path": f"/tmp/frame_{index:03d}.jpg",
                "selection_reason": "selected for temporal coverage",
                "score": index / 10,
            }
            for index in range(1, 8)
        ]
        visual_result = {
            "evidence_frames": evidence_frames,
            "recommended_frames_for_verdict_agent": [{"frame_id": "frame_001", "reason": "recommended"}],
            "high_contact_frame_ids": ["frame_006"],
            "contact_points": [{"frame_id": "frame_005", "target_body_area": "head", "description": "Head contact."}],
            "issue_cards": [{"issue_type": "high_contact", "frame_ids": ["frame_004"], "description": "High contact."}],
        }

        with patch.dict("os.environ", {"REFCHECK_MAX_VERDICT_FRAMES": "3"}, clear=False):
            frames = services._pick_verdict_frames(visual_result)

        self.assertEqual(len(frames), 3)
        self.assertEqual({frame["frame_id"] for frame in frames}, {"frame_004", "frame_005", "frame_006"})

    def test_verdict_frame_picker_prefers_goaltending_evidence_when_selected_call_is_goaltending(self):
        evidence_frames = [
            {
                "frame_id": "frame_001",
                "timestamp": "0:01.00",
                "local_path": "/tmp/frame_001.jpg",
                "selection_reason": "before contact window",
                "score": 0.3,
            },
            {
                "frame_id": "frame_002",
                "timestamp": "0:02.00",
                "local_path": "/tmp/frame_002.jpg",
                "selection_reason": "likely contact window",
                "score": 0.9,
            },
            {
                "frame_id": "frame_003",
                "timestamp": "0:03.00",
                "local_path": "/tmp/frame_003.jpg",
                "selection_reason": "after contact window",
                "score": 0.6,
            },
        ]
        visual_result = {
            "original_call": "Goaltending",
            "primary_issue": "goaltending",
            "evidence_frames": evidence_frames,
            "recommended_frames_for_verdict_agent": [
                {"frame_id": "frame_001", "reason": "ball trajectory before rim contact"},
                {"frame_id": "frame_003", "reason": "cylinder position after touch"},
            ],
            "issue_cards": [{"issue_type": "goaltending", "frame_ids": ["frame_003"], "description": "Goaltending check."}],
            "contact_points": [{"frame_id": "frame_002", "target_body_area": "torso", "description": "Body contact."}],
            "high_contact_frame_ids": ["frame_002"],
        }

        with patch.dict("os.environ", {"REFCHECK_MAX_VERDICT_FRAMES": "3"}, clear=False):
            frames = services._pick_verdict_frames(visual_result)

        self.assertEqual(len(frames), 3)
        selected_ids = {frame["frame_id"] for frame in frames}
        self.assertIn("frame_001", selected_ids)
        self.assertIn("frame_003", selected_ids)


class GeminiResilienceConfigTests(SimpleTestCase):
    def test_retryable_error_detection_catches_known_signals(self):
        self.assertTrue(services._is_retryable_gemini_error(Exception("503 UNAVAILABLE high demand")))
        self.assertTrue(services._is_retryable_gemini_error(Exception("RESOURCE_EXHAUSTED")))
        self.assertTrue(services._is_retryable_gemini_error(Exception("request timeout")))
        self.assertFalse(services._is_retryable_gemini_error(Exception("invalid argument")))

    def test_model_env_resolution(self):
        env = {
            "GEMINI_VISUAL_MODEL": "gemini-2.5-flash-lite",
            "GEMINI_VERDICT_MODEL": "gemini-2.5-pro",
            "GEMINI_FALLBACK_MODEL": "gemini-2.5-flash",
            "GEMINI_MAX_RETRIES": "3",
        }
        with patch.dict("os.environ", env, clear=False):
            self.assertEqual(services._gemini_visual_model(), "gemini-2.5-flash-lite")
            self.assertEqual(services._gemini_verdict_model(), "gemini-2.5-pro")
            self.assertEqual(services._gemini_fallback_model(), "gemini-2.5-flash")
            self.assertEqual(services._gemini_max_retries(), 3)

    def test_verdict_payload_forces_inconclusive_for_goaltending_without_ball_evidence(self):
        result = services._new_verdict_result("test-model")
        payload = {
            "verdict": "Fair Call",
            "confidence": "High",
            "reasoning": "Contact appears visible.",
            "selected_call_reasoning": "Body contact is present.",
            "issue_analysis": [],
            "key_factors": [],
            "limitations": [],
        }
        visual_result = {
            "can_reason_about_call": False,
            "missing_evidence": ["Exact frame of defender-ball contact."],
            "confidence_in_visual_description": "Medium",
            "possible_high_contact": False,
        }

        services._apply_verdict_payload(
            result=result,
            payload=payload,
            frame_id_map={},
            matched_rules_by_issue={},
            original_call="Goaltending",
            visual_result=visual_result,
        )

        self.assertEqual(result["verdict"], "Inconclusive")
        self.assertEqual(result["confidence"], "Medium")
