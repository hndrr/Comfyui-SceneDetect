import importlib
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

text_node = (
    importlib.import_module(f"{video_node.__package__}.pyscenedetect_scene_text")
    if video_node is not None else None
)


class SceneTextTests(unittest.TestCase):
    def test_default_and_custom_prompts(self):
        rows = timecodes_to_dict([(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))], 10.0)
        text, prompts = format_scenes_for_llm(rows)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(text, prompts[0])
        self.assertIn("Scene 1/1:", prompts[0])
        self.assertIn("Describe this shot.", prompts[0])
        combined, custom = format_scenes_for_llm(rows, "Shot {index}/{scene_count} {duration_sec:.3f}s {clip_path} {unknown}")
        self.assertEqual(custom, ["Shot 1/1 2.000s  {unknown}"])
        self.assertEqual(combined, "Shot 1/1 2.000s  {unknown}")

    def test_combined_prompts_reflect_template_and_preserve_scene_order(self):
        rows = timecodes_to_dict([
            (FrameTimecode(0, 10.0), FrameTimecode(20, 10.0)),
            (FrameTimecode(20, 10.0), FrameTimecode(50, 10.0)),
        ], 10.0)
        combined, prompts = format_scenes_for_llm(
            rows, "Shot {index}/{scene_count}: {duration_sec:.1f}s\nDescribe the image.",
        )
        self.assertEqual(prompts, [
            "Shot 1/2: 2.0s\nDescribe the image.",
            "Shot 2/2: 3.0s\nDescribe the image.",
        ])
        self.assertEqual(combined, "Shot 1/2: 2.0s\nDescribe the image.\nShot 2/2: 3.0s\nDescribe the image.")
        self.assertEqual(format_scenes_for_llm(rows, "Scene {index}"), ("Scene 1\nScene 2", ["Scene 1", "Scene 2"]))

    def test_empty_scenes_and_malformed_template(self):
        self.assertEqual(format_scenes_for_llm([]), ("", []))
        rows = timecodes_to_dict([(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))], 10.0)
        self.assertEqual(format_scenes_for_llm(rows, "Unclosed {index"), ("Unclosed {index", ["Unclosed {index"]))
        for template in ("Unknown {missing.key}", "Bad index {index[0]}", "Unknown {missing[key]}"):
            with self.subTest(template=template):
                self.assertEqual(format_scenes_for_llm(rows, template)[1], [template])


    @unittest.skipIf(InputImpl is None, "ComfyUI's comfy_api is not available")
    def test_scene_text_accepts_json_from_both_detectors_and_respects_limits(self):
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
                for limit in (0, 1):
                    with self.subTest(node=cls.__name__, limit=limit):
                        result = cls().run(
                            *args, method="content", threshold=10.0, min_scene_len_sec=0.0,
                            min_scene_len_frames=1, luma_only=False, limit_scenes=limit,
                        )
                        images, scenes_json, count = result[:3]
                        text, prompts = text_node.PySceneDetectSceneText.execute(
                            scenes_json, "Scene {index}/{scene_count}: {duration_frames} frames",
                        ).result
                        expected_count = limit or 2
                        self.assertEqual(images.shape, (expected_count, 32, 32, 3))
                        self.assertEqual(count, expected_count)
                        self.assertEqual(len(json.loads(scenes_json)["scenes"]), count)
                        self.assertEqual(text, "\n".join(f"Scene {i}/{count}: 20 frames" for i in range(1, count + 1)))
                        self.assertEqual(prompts, [f"Scene {i}/{count}: 20 frames" for i in range(1, count + 1)])
                        self.assertNotIn("prompt_template", cls.INPUT_TYPES()["optional"])
                        self.assertNotIn("all_scenes_text", cls.RETURN_NAMES)
                        self.assertNotIn("all_scene_prompts", cls.RETURN_NAMES)
                        self.assertNotIn("per_scene_prompt_list", cls.RETURN_NAMES)
                        self.assertEqual(len(result), 4 if cls is video_node.PySceneDetectVideo else 3)

    @unittest.skipIf(text_node is None, "ComfyUI's comfy_api is not available")
    def test_scene_text_schema_and_default_template(self):
        cls = text_node.PySceneDetectSceneText
        schema = cls.define_schema()
        self.assertEqual(schema.node_id, "PySceneDetectSceneText")
        self.assertEqual(schema.display_name, "PySceneDetect: Scene Text")
        self.assertEqual([item.id for item in schema.inputs], ["scenes_json", "prompt_template"])
        self.assertTrue(schema.inputs[0].force_input)
        self.assertEqual([item.id for item in schema.outputs], ["all_scene_prompts", "per_scene_prompt_list"])
        self.assertEqual([item.is_output_list for item in schema.outputs], [False, True])
        rows = timecodes_to_dict([(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))], 10.0)
        scenes_json = json.dumps({"scenes": rows})
        expected = format_scenes_for_llm(rows)
        self.assertEqual(cls.execute(scenes_json).result, expected)
        self.assertEqual(cls.execute(scenes_json, schema.inputs[1].default).result, expected)
        self.assertEqual(cls.execute(scenes_json, "Unclosed {index").result[1], ["Unclosed {index"])

    @unittest.skipIf(text_node is None, "ComfyUI's comfy_api is not available")
    def test_scene_text_rejects_invalid_metadata(self):
        for value in ("", "{", "[]", "{}", '{"scenes": null}', '{"scenes": [null]}', '{"scenes": [{}]}'):
            with self.subTest(scenes_json=value), self.assertRaises(ValueError):
                text_node.PySceneDetectSceneText.execute(value)


if __name__ == "__main__":
    unittest.main()
