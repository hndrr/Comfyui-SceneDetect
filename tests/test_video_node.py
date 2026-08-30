import importlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from comfy_api.latest import InputImpl


video_node = importlib.import_module(
    "custom_nodes.Comfyui-SceneDetect.nodes.pyscenedetect_video"
)


class VideoNodeTests(unittest.TestCase):
    def test_official_video_input_finds_scenes_and_frames(self):
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

            images, scenes_json, count = video_node.PySceneDetectVideo().run(
                InputImpl.VideoFromFile(str(video_path)),
                method="content",
                threshold=10.0,
                min_scene_len_sec=0.0,
                min_scene_len_frames=1,
                luma_only=False,
            )

        scenes = json.loads(scenes_json)["scenes"]
        self.assertEqual(count, 2)
        self.assertEqual(images.shape, (2, 32, 32, 3))
        self.assertEqual(
            [(scene["start_frame"], scene["end_frame"]) for scene in scenes],
            [(0, 20), (20, 40)],
        )


if __name__ == "__main__":
    unittest.main()
