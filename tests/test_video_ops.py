import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from scenedetect import FrameTimecode, is_ffmpeg_available
from scenedetect.detectors import (
    AdaptiveDetector,
    ContentDetector,
    HashDetector,
    HistogramDetector,
    ThresholdDetector,
)

from utils.video_ops import (
    DetectorSettings,
    TensorVideoStream,
    choose_detector,
    detect_scenes,
    detect_scenes_from_video,
    format_scenes_for_llm,
    normalized_kernel_size,
    read_video_frames,
    sanitize_clip_name,
    split_scene_clips,
    timecodes_to_dict,
    unpack_method_input,
    unpack_toggle_combo,
)


def _write_hard_cut(path: Path, frames_per_scene: int = 20) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 32),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    try:
        for value in (0, 255):
            frame = np.full((32, 32, 3), value, dtype=np.uint8)
            for _ in range(frames_per_scene):
                writer.write(frame)
    finally:
        writer.release()


class VideoOpsTests(unittest.TestCase):
    def test_tensor_video_stream_reads_normalized_bhwc_as_bgr(self):
        frames = torch.zeros((2, 8, 12, 3), dtype=torch.float32)
        frames[0, :, :, 0] = 1.0
        video = TensorVideoStream(frames, 29.97)

        frame = video.read()

        self.assertIsInstance(frame, np.ndarray)
        np.testing.assert_array_equal(frame[0, 0], [0, 0, 255])
        self.assertEqual(frame.shape, (8, 12, 3))
        self.assertEqual(video.frame_number, 1)
        self.assertEqual(video.position.frame_num, 0)
        self.assertEqual(video.duration.frame_num, 2)

    def test_tensor_video_stream_reads_bchw_without_copying_batch(self):
        frames = torch.zeros((2, 3, 8, 12), dtype=torch.uint8)
        frames[0, 1, :, :] = 255
        video = TensorVideoStream(frames, 24.0)

        frame = video.frame_at(0)

        np.testing.assert_array_equal(frame[0, 0], [0, 255, 0])
        self.assertEqual(
            video._frames.untyped_storage().data_ptr(),
            frames.untyped_storage().data_ptr(),
        )

    def test_timecodes_to_dict_uses_v07_properties(self):
        rows = timecodes_to_dict(
            [(FrameTimecode(10, 10.0), FrameTimecode(25, 10.0))],
            10.0,
        )

        self.assertEqual(rows[0]["start_frame"], 10)
        self.assertEqual(rows[0]["end_frame"], 25)
        self.assertEqual(rows[0]["duration_frames"], 15)
        self.assertEqual(rows[0]["duration_sec"], 1.5)
        self.assertEqual(rows[0]["start_time"], "00:00:01.000")

    def test_detect_scenes_preserves_seconds_for_vfr_video(self):
        video = Mock(frame_rate=Fraction(24000, 1001))
        manager = Mock()
        manager.get_scene_list.return_value = []
        detector = object()

        with (
            patch("utils.video_ops.open_video", return_value=video),
            patch("utils.video_ops.SceneManager", return_value=manager),
            patch("utils.video_ops.choose_detector", return_value=detector) as choose,
        ):
            detect_scenes(
                "vfr.mp4",
                method="content",
                threshold=27.0,
                min_scene_len_sec=1.25,
                min_scene_len_frames=300,
                luma_only=False,
            )

        choose.assert_called_once_with(
            "content", 27.0, 1.25, False, settings=DetectorSettings()
        )
        manager.get_scene_list.assert_called_once_with(start_in_scene=False)

    def test_detect_scenes_uses_frames_when_seconds_are_zero(self):
        video = Mock(frame_rate=Fraction(24, 1))
        manager = Mock()
        manager.get_scene_list.return_value = []
        detector = object()
        settings = DetectorSettings(downscale=2, start_in_scene=True)

        with (
            patch("utils.video_ops.SceneManager", return_value=manager),
            patch("utils.video_ops.choose_detector", return_value=detector) as choose,
        ):
            detect_scenes_from_video(
                video,
                method="content",
                threshold=27.0,
                min_scene_len_sec=0.0,
                min_scene_len_frames=15,
                luma_only=False,
                settings=settings,
            )

        choose.assert_called_once_with(
            "content", 27.0, 15, False, settings=settings
        )
        self.assertFalse(manager.auto_downscale)
        self.assertEqual(manager.downscale, 2)
        manager.get_scene_list.assert_called_once_with(start_in_scene=True)

    def test_choose_detector_selects_method_specific_classes(self):
        settings = DetectorSettings(
            adaptive_threshold=4.5,
            hash_threshold=0.2,
            hist_threshold=0.1,
            threshold_method="ceiling",
            fade_bias=0.5,
            add_final_scene=True,
        )

        content = choose_detector("content", 27.0, 15, True, settings)
        adaptive = choose_detector("adaptive", 27.0, 15, True, settings)
        threshold = choose_detector("threshold", 12.0, 15, True, settings)
        hashed = choose_detector("hash", 27.0, 15, True, settings)
        histogram = choose_detector("histogram", 27.0, 15, True, settings)

        self.assertIsInstance(content, ContentDetector)
        self.assertIsInstance(adaptive, AdaptiveDetector)
        self.assertIsInstance(threshold, ThresholdDetector)
        self.assertIsInstance(hashed, HashDetector)
        self.assertIsInstance(histogram, HistogramDetector)
        self.assertEqual(adaptive.adaptive_threshold, 4.5)
        self.assertEqual(hashed._threshold, 0.2)
        self.assertEqual(histogram._bins, 256)
        self.assertEqual(threshold.method, ThresholdDetector.Method.CEILING)
        self.assertEqual(threshold.fade_bias, 0.5)
        self.assertTrue(threshold.add_final_scene)
        with self.assertRaises(ValueError):
            choose_detector("unknown", 27.0, 15, False)

    def test_normalized_kernel_size_uses_odd_values(self):
        self.assertIsNone(normalized_kernel_size(0))
        self.assertIsNone(normalized_kernel_size(2))
        self.assertEqual(normalized_kernel_size(3), 3)
        self.assertEqual(normalized_kernel_size(4), 5)

    def test_detect_scenes_finds_hard_cut(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            _write_hard_cut(video_path)
            scenes, fps = detect_scenes(
                str(video_path),
                method="content",
                threshold=10.0,
                min_scene_len_sec=0.0,
                min_scene_len_frames=1,
                luma_only=False,
            )
            selected_frames = read_video_frames(str(video_path), [0, 20])

        self.assertAlmostEqual(fps, 10.0)
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in scenes],
            [(0, 20), (20, 40)],
        )
        np.testing.assert_array_equal(selected_frames[0][0, 0], [0, 0, 0])
        self.assertGreater(int(selected_frames[20][0, 0].mean()), 200)

    def test_histogram_finds_hard_cut(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            _write_hard_cut(video_path)
            histogram, _ = detect_scenes(
                str(video_path),
                method="histogram",
                threshold=27.0,
                min_scene_len_sec=0.0,
                min_scene_len_frames=1,
                luma_only=False,
            )

        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in histogram],
            [(0, 20), (20, 40)],
        )

    def test_hash_finds_pattern_cut_in_tensor_stream(self):
        rng = np.random.default_rng(0)
        first = torch.from_numpy(
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ).unsqueeze(0).repeat(20, 1, 1, 1)
        second = torch.from_numpy(
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ).unsqueeze(0).repeat(20, 1, 1, 1)
        video = TensorVideoStream(torch.cat((first, second)).float() / 255.0, 10.0)

        hashed, fps = detect_scenes_from_video(
            video,
            method="hash",
            threshold=27.0,
            min_scene_len_sec=0.0,
            min_scene_len_frames=1,
            luma_only=False,
        )

        self.assertAlmostEqual(fps, 10.0)
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in hashed],
            [(0, 20), (20, 40)],
        )

    def test_detect_scenes_finds_hard_cut_in_tensor_stream(self):
        frames = torch.cat(
            (
                torch.zeros((20, 32, 32, 3), dtype=torch.float32),
                torch.ones((20, 32, 32, 3), dtype=torch.float32),
            )
        )
        video = TensorVideoStream(frames, 10.0)

        scenes, fps = detect_scenes_from_video(
            video,
            method="content",
            threshold=10.0,
            min_scene_len_sec=0.0,
            min_scene_len_frames=1,
            luma_only=False,
        )

        self.assertAlmostEqual(fps, 10.0)
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in scenes],
            [(0, 20), (20, 40)],
        )

    def test_format_scenes_for_llm_uses_template_and_defaults(self):
        rows = timecodes_to_dict(
            [(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))],
            10.0,
        )
        rows[0]["clip_path"] = "/tmp/clip.mp4"

        text, prompts = format_scenes_for_llm(rows)
        custom, custom_prompts = format_scenes_for_llm(
            rows, "Shot {index}/{scene_count} {clip_path} {unknown}"
        )

        self.assertIn("# Scenes (1)", text)
        self.assertIn("frames 0-20", text)
        self.assertEqual(len(prompts), 1)
        self.assertIn("Scene 1/1:", prompts[0])
        self.assertIn("Describe this shot.", prompts[0])
        self.assertEqual(custom_prompts[0], "Shot 1/1 /tmp/clip.mp4 {unknown}")
        self.assertIn("# Scenes (1)", custom)

    def test_sanitize_clip_name_strips_unsafe_characters(self):
        self.assertEqual(sanitize_clip_name("My Video (1).mov"), "My_Video_1")
        self.assertEqual(sanitize_clip_name(""), "scene")

    @unittest.skipUnless(is_ffmpeg_available(), "ffmpeg is required to split clips")
    def test_split_scene_clips_writes_one_file_per_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            output_dir = Path(tmpdir) / "clips"
            _write_hard_cut(video_path)
            scenes, _ = detect_scenes(
                str(video_path),
                method="content",
                threshold=10.0,
                min_scene_len_sec=0.0,
                min_scene_len_frames=1,
                luma_only=False,
            )
            clips = split_scene_clips(
                str(video_path),
                scenes,
                str(output_dir),
                video_name="hard-cut",
                reencode=True,
            )

            self.assertEqual(len(clips), 2)
            for clip in clips:
                self.assertTrue(Path(clip).is_file())
                self.assertGreater(Path(clip).stat().st_size, 0)

    def test_unpack_method_input_passes_through_a_plain_name(self):
        method, threshold, luma_only, extra = unpack_method_input(
            "hash",
            threshold=27.0,
            luma_only=True,
            extra={"hash_threshold": 0.2},
        )

        self.assertEqual(method, "hash")
        self.assertEqual(threshold, 27.0)
        self.assertTrue(luma_only)
        self.assertEqual(extra, {"hash_threshold": 0.2})

    def test_unpack_method_input_flattens_a_dynamic_combo_dict(self):
        method, threshold, luma_only, extra = unpack_method_input(
            {
                "method": "content",
                "threshold": 12.5,
                "luma_only": False,
                "delta_hue": 2.0,
                "kernel_size": 5,
            },
            threshold=27.0,
            luma_only=True,
        )

        self.assertEqual(method, "content")
        self.assertEqual(threshold, 12.5)
        self.assertFalse(luma_only)
        self.assertEqual(extra, {"delta_hue": 2.0, "kernel_size": 5})

    def test_unpack_method_input_keeps_hash_threshold_out_of_content_threshold(self):
        method, threshold, luma_only, extra = unpack_method_input(
            {
                "method": "hash",
                "hash_threshold": 0.4,
                "hash_size": 8,
                "hash_lowpass": 2,
            }
        )

        self.assertEqual(method, "hash")
        self.assertEqual(threshold, 27.0)
        self.assertTrue(luma_only)
        self.assertEqual(extra, {"hash_threshold": 0.4, "hash_size": 8, "hash_lowpass": 2})

    def test_unpack_toggle_combo_expands_nested_node_settings(self):
        enabled, extras = unpack_toggle_combo(
            {
                "show_all_settings": "true",
                "max_width": 640,
                "limit_scenes": 4,
                "write_thumbs": True,
                "thumbs_dir": "scene_thumbs",
                "prompt_template": "Scene {index}",
                "start_in_scene": True,
                "downscale": 2,
            },
            "show_all_settings",
        )

        self.assertTrue(enabled)
        self.assertEqual(
            extras,
            {
                "max_width": 640,
                "limit_scenes": 4,
                "write_thumbs": True,
                "thumbs_dir": "scene_thumbs",
                "prompt_template": "Scene {index}",
                "start_in_scene": True,
                "downscale": 2,
            },
        )

    def test_unpack_toggle_combo_false_has_no_nested_fields(self):
        enabled, extras = unpack_toggle_combo("false", "show_all_settings")

        self.assertFalse(enabled)
        self.assertEqual(extras, {})

    def test_unpack_method_input_drops_v1_show_all_settings_flag(self):
        method, threshold, luma_only, extra = unpack_method_input(
            "content",
            extra={"show_all_settings": False, "delta_hue": 2.0},
        )

        self.assertEqual(method, "content")
        self.assertEqual(threshold, 27.0)
        self.assertTrue(luma_only)
        self.assertEqual(extra, {"delta_hue": 2.0})


if __name__ == "__main__":
    unittest.main()
