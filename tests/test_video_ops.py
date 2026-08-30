import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scenedetect import FrameTimecode

from utils.video_ops import detect_scenes, timecodes_to_dict


class VideoOpsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
