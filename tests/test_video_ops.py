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
    TensorVideoStream,
    detect_scenes,
    detect_scenes_from_video,
    timecodes_to_dict,
)


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

        choose.assert_called_once_with("content", 27.0, 1.25, False)

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

        self.assertAlmostEqual(fps, 10.0)
        self.assertEqual(
            [(start.frame_num, end.frame_num) for start, end in scenes],
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


if __name__ == "__main__":
    unittest.main()
