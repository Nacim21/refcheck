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
            "sequence_interpretation": "uncertain_continuity",
            "sequence_interpretation_reason": "The exact defender-ball contact moment is not visible.",
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
        self.assertIn(result["sequence_interpretation"], {"single_play", "uncertain_continuity"})
        self.assertTrue(result["issue_cards"])
        self.assertEqual(result["issue_cards"][0]["issue_type"], "goaltending")
        self.assertTrue(result["secondary_issues"])
        self.assertIn("secondary", result["secondary_issues"][0]["reason"].lower())
        self.assertTrue(any("downward flight" in item.lower() for item in result["missing_evidence"]))


class CoverageFrameSelectionTests(SimpleTestCase):
    def _candidate(
        self,
        candidate_id,
        frame_idx,
        fps=10.0,
        motion=0.1,
        sharpness=100.0,
        relevance=0.5,
        reason="temporal coverage",
    ):
        return {
            "candidate_id": candidate_id,
            "frame_idx": frame_idx,
            "timestamp_seconds": frame_idx / fps,
            "timestamp": services._format_timestamp(frame_idx / fps),
            "motion": motion,
            "sharpness": sharpness,
            "duplicate_similarity": 0.0,
            "relevance": relevance,
            "lower_body_cluster_score": 0.1,
            "floor_activity_score": 0.1,
            "player_heap_score": 0.1,
            "selection_reason": reason,
            "phase_bucket": services._selection_phase(reason),
        }

    def test_candidate_indices_include_boundaries_and_uniform_anchors_for_short_clip(self):
        with patch.dict("os.environ", {"REFCHECK_MAX_CANDIDATE_FRAMES": "20", "REFCHECK_SCAN_FPS": "3"}, clear=False):
            indices = services._candidate_indices(total_frames=90, fps=30.0, duration=3.0)

        self.assertIn(0, indices)
        self.assertIn(89, indices)
        self.assertIn(int(round(89 * 0.50)), indices)
        self.assertTrue(any(idx <= int(89 * 0.08) for idx in indices))
        self.assertTrue(any(idx >= int(89 * 0.92) for idx in indices))

    def test_candidate_indices_include_boundaries_for_long_clip(self):
        with patch.dict("os.environ", {"REFCHECK_MAX_CANDIDATE_FRAMES": "30", "REFCHECK_SCAN_FPS": "4"}, clear=False):
            indices = services._candidate_indices(total_frames=1800, fps=30.0, duration=60.0)

        self.assertIn(0, indices)
        self.assertIn(1799, indices)
        self.assertLessEqual(len(indices), 30)
        self.assertTrue(any(idx <= int(1799 * 0.08) for idx in indices))
        self.assertTrue(any(idx >= int(1799 * 0.92) for idx in indices))

    def test_phase_balanced_selection_keeps_before_contact_and_after(self):
        candidates = [
            self._candidate(1, 0, reason="clip start context", motion=0.02, sharpness=300),
            self._candidate(2, 20, reason="pre-contact context", motion=0.25, sharpness=250),
            self._candidate(3, 30, reason="likely contact moment", motion=0.95, sharpness=220),
            self._candidate(4, 40, reason="post-contact context", motion=0.45, sharpness=230),
            self._candidate(5, 60, reason="aftermath context", motion=0.15, sharpness=200),
            self._candidate(6, 99, reason="clip end context", motion=0.02, sharpness=300),
        ]

        selected = services._select_evidence_candidates(candidates, target_min=4, target_max=6)
        phases = {item["phase_bucket"] for item in selected}

        self.assertIn("pre_contact", phases)
        self.assertIn("likely_contact", phases)
        self.assertIn("post_contact", phases)

    def test_quiet_sharp_boundary_context_survives_selection(self):
        candidates = [
            self._candidate(1, 0, reason="clip start context", motion=0.01, sharpness=500, relevance=0.9),
            self._candidate(2, 10, reason="temporal coverage", motion=0.2, sharpness=90),
            self._candidate(3, 20, reason="likely contact moment", motion=1.0, sharpness=100),
            self._candidate(4, 30, reason="post-contact context", motion=0.5, sharpness=120),
            self._candidate(5, 40, reason="temporal coverage", motion=0.6, sharpness=90),
            self._candidate(6, 99, reason="clip end context", motion=0.01, sharpness=500, relevance=0.9),
        ]

        selected = services._select_evidence_candidates(candidates, target_min=4, target_max=4)
        selected_ids = {item["candidate_id"] for item in selected}

        self.assertIn(1, selected_ids)
        self.assertIn(6, selected_ids)

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

    def test_new_visual_debug_uses_coverage_selection_fields(self):
        debug = services._new_visual_result("test-model")["debug"]

        self.assertIn("boundary_candidates_count", debug)
        self.assertIn("phase_bucket_counts", debug)
        self.assertIn("phase_distribution_selected", debug)
        self.assertIn("selection_mode", debug)
        self.assertNotIn("adaptive_frame_selection", debug)

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
                "selection_reason": "pre-contact context",
                "score": 0.3,
            },
            {
                "frame_id": "frame_002",
                "timestamp": "0:02.00",
                "local_path": "/tmp/frame_002.jpg",
                "selection_reason": "likely contact moment",
                "score": 0.9,
            },
            {
                "frame_id": "frame_003",
                "timestamp": "0:03.00",
                "local_path": "/tmp/frame_003.jpg",
                "selection_reason": "post-contact context",
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

    def test_verdict_frame_picker_preserves_phase_spread_under_tight_cap(self):
        evidence_frames = [
            {
                "frame_id": "frame_001",
                "timestamp": "0:01.00",
                "local_path": "/tmp/frame_001.jpg",
                "selection_reason": "pre-contact context",
                "phase_bucket": "pre_contact",
                "score": 0.6,
            },
            {
                "frame_id": "frame_002",
                "timestamp": "0:02.00",
                "local_path": "/tmp/frame_002.jpg",
                "selection_reason": "likely contact moment",
                "phase_bucket": "likely_contact",
                "score": 0.9,
            },
            {
                "frame_id": "frame_003",
                "timestamp": "0:03.00",
                "local_path": "/tmp/frame_003.jpg",
                "selection_reason": "post-contact context",
                "phase_bucket": "post_contact",
                "score": 0.7,
            },
            {
                "frame_id": "frame_004",
                "timestamp": "0:02.10",
                "local_path": "/tmp/frame_004.jpg",
                "selection_reason": "likely contact moment",
                "phase_bucket": "likely_contact",
                "score": 0.95,
            },
        ]
        visual_result = {
            "evidence_frames": evidence_frames,
            "recommended_frames_for_verdict_agent": [
                {"frame_id": "frame_002", "reason": "contact"},
                {"frame_id": "frame_004", "reason": "contact"},
                {"frame_id": "frame_001", "reason": "before"},
                {"frame_id": "frame_003", "reason": "after"},
            ],
        }

        with patch.dict("os.environ", {"REFCHECK_MAX_VERDICT_FRAMES": "3"}, clear=False):
            frames = services._pick_verdict_frames(visual_result)

        phases = {frame["phase_bucket"] for frame in frames}
        self.assertEqual(len(frames), 3)
        self.assertEqual(phases, {"pre_contact", "likely_contact", "post_contact"})


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


class VerdictAgentPipelineTests(SimpleTestCase):
    def test_verdict_prompt_includes_agent_1_description_frames_and_rules(self):
        frames = [
            {
                "frame_id": "frame_001",
                "timestamp": "0:01.00",
                "local_path": "/tmp/frame_001.jpg",
                "reason": "pre-contact context",
            },
            {
                "frame_id": "frame_002",
                "timestamp": "0:02.00",
                "local_path": "/tmp/frame_002.jpg",
                "reason": "likely contact moment",
            },
        ]
        prompt = services._build_verdict_prompt(
            "White offense drives and dark defender contests at the rim.",
            frames,
            original_call="Goaltending",
            matched_rules=[{"rule_id": "rule-summary", "section": "Rule 11", "text": "Goaltending rule text."}],
            matched_rules_by_issue={
                "selected_call_goaltending": [
                    {"rule_id": "rule-gt", "section": "Rule 11-I", "text": "A player may not touch the ball on its downward flight."}
                ]
            },
            officiating_issue_summary="Review whether the defender touched the ball after it began descending.",
            key_events=[
                {
                    "frame_id": "frame_002",
                    "timestamp": "0:02.00",
                    "description": "The defender reaches toward the ball near the rim.",
                }
            ],
            possible_critical_moments=[
                {
                    "frame_id": "frame_002",
                    "timestamp": "0:02.00",
                    "description": "Potential ball touch near the rim.",
                    "why_relevant": "possible contact",
                }
            ],
            issue_cards=[
                {
                    "issue_id": "issue_001",
                    "issue_type": "goaltending",
                    "description": "Determine downward flight and ball touch timing.",
                    "frame_ids": ["frame_002"],
                }
            ],
        )

        self.assertIn("Visual Analyst summary (verbatim):", prompt)
        self.assertIn("Agent 1 neutral officiating issue summary:", prompt)
        self.assertIn("Review whether the defender touched the ball after it began descending.", prompt)
        self.assertIn("Frame-bound key events from Agent 1:", prompt)
        self.assertIn("The defender reaches toward the ball near the rim.", prompt)
        self.assertIn("Possible critical moments from Agent 1:", prompt)
        self.assertIn("Curated frames for verdict review (in temporal order):", prompt)
        self.assertIn("frame_001 at 0:01.00", prompt)
        self.assertIn("Rulebook context grouped by issue:", prompt)
        self.assertIn("selected_call_goaltending", prompt)
        self.assertIn("Rule 11-I", prompt)
        self.assertIn("relevant_rules_used", prompt)
        self.assertIn("retrieved database rules", prompt)

    def test_run_verdict_agent_sends_curated_frames_agent_1_context_and_rules(self):
        visual_result = {
            "status": "Analyzed",
            "video_summary": "White offense drives and dark defender contests at the rim.",
            "original_call": "Goaltending",
            "evidence_frames": [
                {
                    "frame_id": "frame_001",
                    "timestamp": "0:01.00",
                    "local_path": "/tmp/frame_001.jpg",
                    "selection_reason": "pre-contact context",
                    "phase_bucket": "pre_contact",
                    "score": 0.6,
                },
                {
                    "frame_id": "frame_002",
                    "timestamp": "0:02.00",
                    "local_path": "/tmp/frame_002.jpg",
                    "selection_reason": "likely contact moment",
                    "phase_bucket": "likely_contact",
                    "score": 0.9,
                },
            ],
            "recommended_frames_for_verdict_agent": [
                {"frame_id": "frame_001", "reason": "ball trajectory before rim contact"},
                {"frame_id": "frame_002", "reason": "defender-ball touch timing"},
            ],
            "officiating_issue_summary": "Review ball trajectory and defender touch timing.",
            "key_events": [
                {
                    "frame_id": "frame_002",
                    "timestamp": "0:02.00",
                    "description": "The defender reaches toward the ball near the rim.",
                }
            ],
            "possible_critical_moments": [
                {
                    "frame_id": "frame_002",
                    "timestamp": "0:02.00",
                    "description": "Potential ball touch near the rim.",
                    "why_relevant": "possible contact",
                }
            ],
            "issue_cards": [
                {
                    "issue_id": "issue_001",
                    "issue_type": "goaltending",
                    "description": "Determine downward flight and ball touch timing.",
                    "frame_ids": ["frame_002"],
                    "rag_query": "NBA goaltending downward flight ball touch",
                }
            ],
            "contact_points": [],
            "secondary_issues": [],
            "primary_issue": "goaltending",
            "verdict_readiness": "Ready",
            "can_reason_about_call": True,
            "sequence_interpretation": "single_play",
            "confidence_in_visual_description": "Medium",
            "possible_high_contact": False,
            "high_contact_description": "",
            "missing_evidence": [],
        }
        retrieved_rules = {
            "selected_call_goaltending": [
                {
                    "rule_id": "rule-gt",
                    "section": "Rule 11-I",
                    "score": 0.91,
                    "text": "A player may not touch the ball on its downward flight.",
                }
            ]
        }
        captured = {}

        def fake_call(api_key, model_name, prompt, frames_for_verdict, result):
            captured["prompt"] = prompt
            captured["frames"] = frames_for_verdict
            return (
                '{"agent":"verdict_agent","original_call":"Goaltending","verdict":"Inconclusive",'
                '"confidence":"Medium","correct_ruling_if_any":"","reasoning":"The provided frames show a contest near the rim, but the exact ball trajectory remains partly ambiguous.",'
                '"selected_call_reasoning":"The goaltending review depends on downward flight and defender-ball touch timing, which are not fully resolved.",'
                '"other_possible_issues":[],"issue_analysis":[{"issue_id":"issue_001","issue_type":"goaltending","finding":"The frames require analysis of defender-ball touch timing.","confidence":"Medium","relevant_frames":["frame_002"],"relevant_rules":["rule-gt"]}],'
                '"key_factors":[{"frame_id":"frame_002","timestamp":"0:02.00","factor":"The defender reaches toward the ball near the rim."}],'
                '"rule_basis":"Rule 11-I downward-flight goaltending considerations drive the review.",'
                '"relevant_rules_used":[{"issue_id":"issue_001","rule_id":"rule-gt","section":"Rule 11-I","why_relevant":"This retrieved database rule defines the downward-flight goaltending question."}],'
                '"limitations":["Exact ball trajectory is not fully continuous."]}'
            )

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False), patch.object(
            services, "retrieve_rules_for_issues", return_value=retrieved_rules
        ), patch.object(services, "_call_gemini_for_verdict", side_effect=fake_call):
            result = services.run_verdict_agent(visual_result)

        self.assertEqual(result["status"], "Analyzed")
        self.assertEqual(result["matched_rules_by_issue"], retrieved_rules)
        self.assertEqual([frame["frame_id"] for frame in captured["frames"]], ["frame_001", "frame_002"])
        self.assertIn("Review ball trajectory and defender touch timing.", captured["prompt"])
        self.assertIn("The defender reaches toward the ball near the rim.", captured["prompt"])
        self.assertIn("Rulebook context grouped by issue:", captured["prompt"])
        self.assertIn("Rule 11-I", captured["prompt"])
        self.assertEqual(result["relevant_rules_used"][0]["rule_id"], "rule-gt")
        self.assertEqual(result["relevant_rules_used"][0]["section"], "Rule 11-I")
        self.assertIn("retrieved database rule", result["relevant_rules_used"][0]["why_relevant"])

    def test_run_verdict_agent_caps_rules_visible_to_agent_2_at_three_total(self):
        visual_result = {
            "status": "Analyzed",
            "video_summary": "White offense drives and dark defender contests at the rim.",
            "original_call": "Personal foul",
            "evidence_frames": [
                {
                    "frame_id": "frame_001",
                    "timestamp": "0:01.00",
                    "local_path": "/tmp/frame_001.jpg",
                    "selection_reason": "likely contact moment",
                    "phase_bucket": "likely_contact",
                    "score": 0.9,
                }
            ],
            "recommended_frames_for_verdict_agent": [{"frame_id": "frame_001", "reason": "contact"}],
            "issue_cards": [],
            "contact_points": [],
            "secondary_issues": [],
            "primary_issue": "personal_foul",
            "verdict_readiness": "Ready",
            "can_reason_about_call": True,
            "sequence_interpretation": "single_play",
            "confidence_in_visual_description": "Medium",
            "possible_high_contact": False,
            "high_contact_description": "",
            "missing_evidence": [],
        }
        retrieved_rules = {
            "summary": [
                {"rule_id": "rule-1", "section": "Rule 1", "score": 0.99, "text": "First rule."},
                {"rule_id": "rule-2", "section": "Rule 2", "score": 0.98, "text": "Second rule."},
            ],
            "contact_001": [
                {"rule_id": "rule-3", "section": "Rule 3", "score": 0.97, "text": "Third rule."},
                {"rule_id": "rule-4", "section": "Rule 4", "score": 0.96, "text": "Fourth rule."},
                {"rule_id": "rule-5", "section": "Rule 5", "score": 0.95, "text": "Fifth rule."},
            ],
        }
        captured = {}

        def fake_call(api_key, model_name, prompt, frames_for_verdict, result):
            captured["prompt"] = prompt
            return (
                '{"agent":"verdict_agent","original_call":"Personal foul","verdict":"Inconclusive",'
                '"confidence":"Low","correct_ruling_if_any":"","reasoning":"The frame shows contact, but the full context is limited.",'
                '"selected_call_reasoning":"The personal foul review is limited by the single available frame.",'
                '"other_possible_issues":[],"issue_analysis":[],"key_factors":[{"frame_id":"frame_001","timestamp":"0:01.00","factor":"Contact is visible in the selected frame."}],'
                '"rule_basis":"Rule 1 is one retrieved database rule relevant to the contact review.",'
                '"relevant_rules_used":[{"issue_id":"summary","rule_id":"rule-1","section":"Rule 1","why_relevant":"This retrieved database rule is relevant to contact."}],'
                '"limitations":["Only one frame is available."]}'
            )

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False), patch.object(
            services, "retrieve_rules_for_issues", return_value=retrieved_rules
        ), patch.object(services, "_call_gemini_for_verdict", side_effect=fake_call):
            result = services.run_verdict_agent(visual_result)

        self.assertEqual([rule["rule_id"] for rule in result["matched_rules"]], ["rule-1", "rule-2", "rule-3"])
        self.assertIn("Rule 1", captured["prompt"])
        self.assertIn("Rule 2", captured["prompt"])
        self.assertIn("Rule 3", captured["prompt"])
        self.assertNotIn("Rule 4", captured["prompt"])
        self.assertNotIn("Rule 5", captured["prompt"])
