import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from scenedetect import FrameTimecode

from utils.video_ops import (
    DetectorSettings,
    TensorVideoStream,
    choose_detector,
    detect_scenes,
    detect_scenes_from_video,
    read_video_frames,
    timecodes_to_dict,
)


class VideoOpsTests(unittest.TestCase):
    def test_custom_kernel_sizes_match_documented_automatic_range(self):
        for method in ("content", "adaptive"):
            for size, expected in ((0, None), (1, None), (2, None), (3, 3), (4, 5), (5, 5)):
                with self.subTest(method=method, size=size):
                    detector = choose_detector(
                        method, 27.0, 1, False,
                        settings=DetectorSettings(kernel_size=size),
                    )
                    if expected is None:
                        self.assertIsNone(detector._kernel)
                    else:
                        self.assertEqual(detector._kernel.shape, (expected, expected))

    def test_zero_custom_weights_require_luma_only(self):
        settings = DetectorSettings(
            delta_hue=0.0, delta_sat=0.0, delta_lum=0.0, delta_edges=0.0,
        )
        frames = torch.zeros((40, 16, 16, 3))
        frames[20:] = 1.0
        for method in ("content", "adaptive"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "At least one content weight"):
                    detect_scenes_from_video(
                        TensorVideoStream(frames, 10.0), method, 27.0, 0.0, 1, False,
                        settings=settings,
                    )
                scenes, _ = detect_scenes_from_video(
                    TensorVideoStream(frames, 10.0), method, 27.0, 0.0, 1, True,
                    settings=settings,
                )
                self.assertEqual(
                    [(start.frame_num, end.frame_num) for start, end in scenes],
                    [(0, 20), (20, 40)],
                )
        for method in ("threshold", "hash", "histogram"):
            choose_detector(method, 27.0, 1, False, settings=settings)

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
            "content", 27.0, 1.25, False,
            hash_threshold=0.395, hist_threshold=0.05,
            settings=None,
        )

    def test_detect_scenes_uses_frames_when_seconds_are_zero(self):
        video = Mock(frame_rate=Fraction(24, 1))
        manager = Mock()
        manager.get_scene_list.return_value = []
        detector = object()

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
                downscale=2,
            )

        choose.assert_called_once_with(
            "content", 27.0, 15, False,
            hash_threshold=0.395, hist_threshold=0.05,
            settings=None,
        )
        self.assertFalse(manager.auto_downscale)
        self.assertEqual(manager.downscale, 2)

    def test_detect_scenes_finds_hard_cut(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 32),
            )
            self.assertTrue(writer.isOpened())
            try:
                for value in (0, 255):
                    frame = np.full((32, 32, 3), value, dtype=np.uint8)
                    for _ in range(20):
                        writer.write(frame)
            finally:
                writer.release()

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
        np.testing.assert_array_equal(selected_frames[20][0, 0], [255, 255, 255])

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

    def test_no_cuts_returns_entire_tensor_stream(self):
        for method in ("content", "adaptive", "threshold", "hash", "histogram"):
            for frame_count in (1, 218):
                with self.subTest(method=method, frame_count=frame_count):
                    video = TensorVideoStream(
                        torch.full((frame_count, 32, 32, 3), 0.5), 10.0
                    )
                    scenes, fps = detect_scenes_from_video(
                        video,
                        method=method,
                        threshold=27.0,
                        min_scene_len_sec=0.0,
                        min_scene_len_frames=15,
                        luma_only=True,
                    )

                    self.assertAlmostEqual(fps, 10.0)
                    self.assertEqual(
                        [(start.frame_num, end.frame_num) for start, end in scenes],
                        [(0, frame_count)],
                    )

    def test_new_detectors_find_cuts_and_use_their_own_thresholds(self):
        rng = np.random.default_rng(0)
        hash_frames = torch.from_numpy(rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8))
        hist_frames = torch.zeros((2, 64, 64, 3), dtype=torch.uint8)
        hist_frames[0, :, 32:] = 255
        hist_frames[1, :, 32:] = 128

        for method, frames, insensitive in (
            ("hash", hash_frames, {"hash_threshold": 1.0}),
            ("histogram", hist_frames, {"hist_threshold": 0.9}),
        ):
            frames = frames.repeat_interleave(20, dim=0)
            for downscale in (0, 1, 2):
                for options, expected in (
                    ({}, [(0, 20), (20, 40)]),
                    (insensitive, [(0, 40)]),
                ):
                    with self.subTest(method=method, downscale=downscale, options=options):
                        scenes, fps = detect_scenes_from_video(
                            TensorVideoStream(frames, 10.0),
                            method=method,
                            threshold=1000.0,
                            min_scene_len_sec=0.0,
                            min_scene_len_frames=1,
                            luma_only=True,
                            downscale=downscale,
                            **options,
                        )
                        self.assertAlmostEqual(fps, 10.0)
                        self.assertEqual(
                            [(start.frame_num, end.frame_num) for start, end in scenes],
                            expected,
                        )

    def test_threshold_details_control_fade_boundaries(self):
        frames = torch.cat((
            torch.ones((20, 32, 32, 3)),
            torch.zeros((10, 32, 32, 3)),
            torch.ones((20, 32, 32, 3)),
        ))
        for bias, boundary in ((-1.0, 20), (0.0, 25), (1.0, 30)):
            with self.subTest(bias=bias):
                scenes, _ = detect_scenes_from_video(
                    TensorVideoStream(frames, 10.0),
                    "threshold", 12.0, 0.0, 1, True,
                    settings=DetectorSettings(fade_bias=bias),
                )
                self.assertEqual(
                    [(start.frame_num, end.frame_num) for start, end in scenes],
                    [(0, boundary), (boundary, 50)],
                )

        scenes, _ = detect_scenes_from_video(
            TensorVideoStream(frames[:30], 10.0),
            "threshold", 12.0, 0.0, 1, True,
            settings=DetectorSettings(add_final_scene=True),
        )
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in scenes],
            [(0, 20), (20, 30)],
        )
        scenes, _ = detect_scenes_from_video(
            TensorVideoStream(1.0 - frames, 10.0),
            "threshold", 240.0, 0.0, 1, True,
            settings=DetectorSettings(threshold_method="ceiling"),
        )
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in scenes],
            [(0, 25), (25, 50)],
        )


if __name__ == "__main__":
    unittest.main()
