import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from scenedetect import FrameTimecode, open_video
from scenedetect.backends import AVAILABLE_BACKENDS
from scenedetect.common import Timecode
from scenedetect.video_stream import SeekError

from video_fixtures import write_timing_video

from utils.video_ops import (
    DetectorSettings,
    TensorVideoStream,
    choose_detector,
    detect_scenes,
    detect_scenes_from_video,
    pick_frame_timecode,
    read_scene_frames,
    timecodes_to_dict,
)


class VideoOpsTests(unittest.TestCase):
    def test_representative_timecodes_preserve_pts(self):
        start = FrameTimecode(Timecode(1001, Fraction(1, 1000)), 30.0)
        end = FrameTimecode(Timecode(2003, Fraction(1, 1000)), 30.0)

        self.assertIs(pick_frame_timecode((start, end), "start"), start)
        middle = pick_frame_timecode((start, end), "middle")
        self.assertAlmostEqual(middle.seconds, 1.502, places=3)
        self.assertEqual(middle.time_base, Fraction(1, 1000))
        self.assertIs(pick_frame_timecode((start, end), "end"), end)

    def test_scene_reads_distinguish_pts_with_the_same_nominal_frame_number(self):
        timecodes = [
            FrameTimecode(Timecode(1000, Fraction(1, 1000)), 30.0),
            FrameTimecode(Timecode(1001, Fraction(1, 1000)), 30.0),
            FrameTimecode(Timecode(1002, Fraction(1, 1000)), 30.0),
        ]
        self.assertEqual(len({timecode.frame_num for timecode in timecodes}), 1)
        scenes = list(zip(timecodes, timecodes[1:] + [timecodes[-1] + 1]))
        decoded = iter(zip(timecodes, (0, 128, 255)))
        video = Mock()

        def read():
            item = next(decoded, None)
            if item is None:
                return False
            video.position, value = item
            return np.full((8, 8, 3), value, dtype=np.uint8)

        video.read.side_effect = read
        with patch("utils.video_ops.open_video", return_value=video) as open_stream:
            frames = read_scene_frames("vfr.mp4", scenes, "end", backend="pyav")

        open_stream.assert_called_once_with("vfr.mp4", backend="pyav")
        self.assertEqual(len(frames), 3)
        for (actual, frame), expected, value in zip(frames, timecodes, (0, 128, 255)):
            self.assertIs(actual, expected)
            self.assertAlmostEqual(frame.mean(), value)

    @unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
    def test_late_scene_reads_skip_the_untrimmed_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "long.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(1000):
                    writer.write(np.full((32, 32, 3), 128, dtype=np.uint8))
            finally:
                writer.release()
            scene = (FrameTimecode(90.0, 10.0), FrameTimecode(95.0, 10.0))
            for backend in ("opencv", "pyav"):
                with self.subTest(backend=backend):
                    video = open_video(path, backend=backend)
                    with (
                        patch("utils.video_ops.open_video", return_value=video),
                        patch.object(video, "read", wraps=video.read) as read,
                    ):
                        frames = read_scene_frames(path, [scene], "end", backend=backend)
                    self.assertLess(read.call_count, 100)
                    position, frame = frames[0]
                    self.assertGreaterEqual(position, scene[0])
                    self.assertLess(position, scene[1])
                    self.assertAlmostEqual(frame.mean(), 128, delta=3)
                    del video

    @unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
    def test_trimmed_vfr_reads_keep_short_and_final_scenes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "short-vfr.mp4")
            write_timing_video(path, vfr=True, short_scenes=True)
            for backend in ("opencv", "pyav"):
                scenes, _ = detect_scenes(path, "content", 10.0, 0.0, 1, True, backend=backend)
                for first in (1, 3):
                    for representative in ("start", "middle", "end"):
                        with self.subTest(backend=backend, first=first, representative=representative):
                            frames = read_scene_frames(path, scenes[first:], representative, backend)
                            self.assertEqual(len(frames), len(scenes[first:]))
                            for (position, frame), (start, end), value in zip(
                                frames, scenes[first:], (0, 255, 128, 30, 200, 70)[first:]
                            ):
                                self.assertGreaterEqual(position, start)
                                self.assertLess(position, end)
                                self.assertAlmostEqual(frame.mean(), value, delta=3)

    def test_scene_reads_restart_after_unsafe_seek(self):
        scene = (FrameTimecode(90.0, 30.0), FrameTimecode(90.1, 30.0))
        for failure in ("overshoot", "eof", "seek_error"):
            with self.subTest(failure=failure):
                video = Mock(is_seekable=True)
                initial = [(91.0, 255)] if failure == "overshoot" else [None]
                if failure == "seek_error":
                    initial = []
                    video.seek.side_effect = SeekError("seek failed")
                decoded = iter(initial + [(0.0, 0), (89.9, 0), (90.0, 128)])

                def read():
                    item = next(decoded, None)
                    if item is None:
                        return False
                    seconds, value = item
                    video.position = FrameTimecode(seconds, 30.0)
                    return np.full((8, 8, 3), value, dtype=np.uint8)

                video.read.side_effect = read
                with patch("utils.video_ops.open_video", return_value=video):
                    frames = read_scene_frames("vfr.mp4", [scene])
                video.seek.assert_called_once_with(89.0)
                video.reset.assert_called_once_with()
                self.assertEqual(frames[0][0], scene[0])
                self.assertEqual(frames[0][1].mean(), 128)

    def test_missing_scene_frame_raises_instead_of_silently_dropping_an_image(self):
        video = Mock()
        video.read.return_value = False
        scene = (FrameTimecode(0, 30.0), FrameTimecode(1, 30.0))
        with patch("utils.video_ops.open_video", return_value=video):
            with self.assertRaisesRegex(ValueError, "No representative frame"):
                read_scene_frames("missing.mp4", [scene])
        self.assertEqual(read_scene_frames("unused.mp4", []), [])

    @unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
    def test_vfr_representatives_use_scene_times_with_both_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vfr.mp4"
            write_timing_video(path, vfr=True)
            for backend in ("opencv", "pyav"):
                with self.subTest(backend=backend):
                    scenes, _ = detect_scenes(
                        str(path), "content", 10.0, 0.0, 1, True,
                        backend=backend,
                    )
                    self.assertEqual(len(scenes), 2)
                    self.assertAlmostEqual(scenes[1][0].seconds, 3.0)
                    for representative in ("start", "middle", "end"):
                        frames = read_scene_frames(str(path), scenes, representative, backend=backend)
                        self.assertLess(frames[0][1].mean(), 5)
                        self.assertGreater(frames[1][1].mean(), 250)
                        for (position, _), (start, end) in zip(frames, scenes):
                            self.assertGreaterEqual(position, start)
                            self.assertLess(position, end)

    def test_backend_fallback_is_left_to_pyscenedetect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "fallback.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(20):
                    writer.write(np.full((32, 32, 3), 128, dtype=np.uint8))
            finally:
                writer.release()

            with patch.dict(AVAILABLE_BACKENDS, {"opencv": AVAILABLE_BACKENDS["opencv"]}, clear=True):
                scenes, fps = detect_scenes(path, "content", 10.0, 0.0, 1, True, backend="pyav")
                frames = read_scene_frames(path, scenes, backend="pyav")

        self.assertEqual([(start.frame_num, end.frame_num) for start, end in scenes], [(0, 20)])
        self.assertAlmostEqual(fps, 10.0)
        self.assertAlmostEqual(frames[0][1].mean(), 128, delta=3)

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

    def test_positive_pts_scenes_survive_equal_nominal_frame_numbers(self):
        times = [FrameTimecode(Timecode(t, Fraction(1, 1000)), 30.0) for t in (1000, 1001, 1002)]
        expected = list(zip(times, times[1:]))
        self.assertEqual(len({time.frame_num for time in times}), 1)
        manager = Mock()
        manager.get_scene_list.return_value = [(times[0], times[0]), *expected]
        with patch("utils.video_ops.SceneManager", return_value=manager):
            scenes, _ = detect_scenes_from_video(
                Mock(frame_rate=Fraction(30)), "content", 10.0, 0.0, 1, True,
            )
        self.assertEqual(scenes, expected)

    def test_detect_scenes_finds_hard_cut(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"FFV1"),  # Preserve exact pixel values across codec builds.
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
            selected_frames = read_scene_frames(str(video_path), scenes)

        self.assertAlmostEqual(fps, 10.0)
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in scenes],
            [(0, 20), (20, 40)],
        )
        np.testing.assert_array_equal(selected_frames[0][1][0, 0], [0, 0, 0])
        np.testing.assert_array_equal(selected_frames[1][1][0, 0], [255, 255, 255])

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
