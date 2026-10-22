import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from scenedetect import FrameTimecode

from utils.scene_text import format_scenes_for_llm
from utils.video_ops import timecodes_to_dict
from test_video_node import InputImpl, legacy_node, video_node


class SceneTextTests(unittest.TestCase):
    def test_default_and_custom_prompts(self):
        rows = timecodes_to_dict([(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))], 10.0)
        text, prompts = format_scenes_for_llm(rows)
        self.assertIn("# Scenes (1)", text)
        self.assertIn("frames 0-20", text)
        self.assertEqual(len(prompts), 1)
        self.assertIn("Scene 1/1:", prompts[0])
        self.assertIn("Describe this shot.", prompts[0])
        _, custom = format_scenes_for_llm(rows, "Shot {index}/{scene_count} {duration_sec:.3f}s {clip_path} {unknown}")
        self.assertEqual(custom, ["Shot 1/1 2.000s  {unknown}"])

    def test_empty_scenes_and_malformed_template(self):
        self.assertEqual(format_scenes_for_llm([]), ("# Scenes (0)", []))
        rows = timecodes_to_dict([(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))], 10.0)
        self.assertEqual(format_scenes_for_llm(rows, "Unclosed {index")[1], ["Unclosed {index"])

    @unittest.skipIf(InputImpl is None, "ComfyUI's comfy_api is not available")
    def test_both_nodes_emit_one_prompt_per_limited_scene(self):
        frames = np.zeros((40, 32, 32, 3), dtype=np.uint8)
        frames[20:] = 255
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hard-cut.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for frame in frames:
                    writer.write(frame)
            finally:
                writer.release()
            for cls, args in (
                (video_node.PySceneDetectVideo, (InputImpl.VideoFromFile(str(path)),)),
                (legacy_node.PySceneDetectToImages, (torch.from_numpy(frames).float() / 255.0, {"loaded_fps": 10.0})),
            ):
                with self.subTest(node=cls.__name__):
                    result = cls().run(
                        *args, method="content", threshold=10.0, min_scene_len_sec=0.0,
                        min_scene_len_frames=1, luma_only=False, limit_scenes=1,
                        prompt_template="Scene {index}/{scene_count}: {duration_frames} frames",
                    )
                    images, scenes_json, count, text, prompts = result
                    self.assertEqual(images.shape, (1, 32, 32, 3))
                    self.assertEqual(count, 1)
                    self.assertEqual(len(json.loads(scenes_json)["scenes"]), count)
                    self.assertEqual(text.splitlines()[0], "# Scenes (1)")
                    self.assertEqual(prompts, ["Scene 1/1: 20 frames"])
                    self.assertEqual(cls.RETURN_NAMES[3:], ("all_scenes_text", "per_scene_prompt_list"))
                    self.assertEqual(cls.OUTPUT_IS_LIST, (False, False, False, False, True))
                    self.assertEqual(list(cls.INPUT_TYPES()["optional"])[-1], "prompt_template")


if __name__ == "__main__":
    unittest.main()
