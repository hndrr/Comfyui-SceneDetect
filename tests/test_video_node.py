import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from comfy_api.latest import InputImpl
except ImportError:
    InputImpl = None


def _load_video_node():
    repository_root = Path(__file__).resolve().parents[1]
    package_name = "comfyui_scenedetect_test_package"

    package = types.ModuleType(package_name)
    package.__path__ = [str(repository_root)]
    sys.modules[package_name] = package

    nodes_package_name = f"{package_name}.nodes"
    nodes_package = types.ModuleType(nodes_package_name)
    nodes_package.__path__ = [str(repository_root / "nodes")]
    sys.modules[nodes_package_name] = nodes_package

    return importlib.import_module(f"{nodes_package_name}.pyscenedetect_video")


video_node = _load_video_node() if InputImpl is not None else None
legacy_node = (
    importlib.import_module(f"{video_node.__package__}.pyscenedetect_to_images")
    if video_node is not None else None
)


@unittest.skipIf(InputImpl is None, "ComfyUI's comfy_api is not available")
class VideoNodeTests(unittest.TestCase):
    def test_thumbnail_path_stays_inside_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = video_node._resolve_thumbnail_path(
                tmpdir, "scene_thumbs/frame.jpg"
            )

        self.assertEqual(
            resolved,
            str(Path(tmpdir).resolve() / "scene_thumbs" / "frame.jpg"),
        )

    def test_thumbnail_path_rejects_absolute_and_traversal_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                video_node._resolve_thumbnail_path(
                    tmpdir, str(Path(tmpdir).parent / "outside")
                )
            with self.assertRaises(ValueError):
                video_node._resolve_thumbnail_path(tmpdir, "../outside")
            with self.assertRaises(ValueError):
                video_node._resolve_thumbnail_path(tmpdir, r"..\outside")
            with self.assertRaises(ValueError):
                video_node._resolve_thumbnail_path(tmpdir, r"C:\outside")

    def test_thumbnail_path_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            outside = Path(tmpdir) / "outside"
            output_root.mkdir()
            outside.mkdir()
            (output_root / "linked").symlink_to(outside, target_is_directory=True)

            with self.assertRaises(ValueError):
                video_node._resolve_thumbnail_path(
                    str(output_root), "linked/frame.jpg"
                )

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

    def test_official_video_without_cuts_returns_scene_and_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "single-scene.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 32),
            )
            self.assertTrue(writer.isOpened())
            try:
                frame = np.full((32, 32, 3), 128, dtype=np.uint8)
                for _ in range(218):
                    writer.write(frame)
            finally:
                writer.release()

            for start_time, duration, start_frame, end_frame in (
                (0.0, 0.0, 0, 218),
                (2.0, 3.0, 20, 50),
            ):
                with self.subTest(start_time=start_time, duration=duration):
                    images, scenes_json, count = video_node.PySceneDetectVideo().run(
                        InputImpl.VideoFromFile(
                            str(video_path), start_time=start_time, duration=duration
                        ),
                        method="content",
                        threshold=27.0,
                        min_scene_len_sec=0.0,
                        min_scene_len_frames=15,
                        luma_only=True,
                    )

                    self.assertEqual(count, 1)
                    self.assertEqual(images.shape, (1, 32, 32, 3))
                    self.assertAlmostEqual(images.mean().item(), 128 / 255, places=3)
                    scene = json.loads(scenes_json)["scenes"][0]
                    self.assertEqual(scene["index"], 1)
                    self.assertEqual(scene["start_frame"], start_frame)
                    self.assertEqual(scene["end_frame"], end_frame)
                    self.assertEqual(scene["duration_frames"], end_frame - start_frame)

    def test_legacy_video_without_cuts_returns_scene_and_frame(self):
        images, scenes_json, count = legacy_node.PySceneDetectToImages().run(
            torch.full((218, 32, 32, 3), 0.5),
            {"loaded_fps": 10.0},
            method="content",
            threshold=27.0,
            min_scene_len_sec=0.0,
            min_scene_len_frames=15,
            luma_only=True,
        )

        self.assertEqual(count, 1)
        self.assertEqual(images.shape, (1, 32, 32, 3))
        self.assertAlmostEqual(images.mean().item(), 127 / 255, places=3)
        scene = json.loads(scenes_json)["scenes"][0]
        self.assertEqual(scene["index"], 1)
        self.assertEqual(scene["start_frame"], 0)
        self.assertEqual(scene["end_frame"], 218)
        self.assertEqual(scene["duration_frames"], 218)


if __name__ == "__main__":
    unittest.main()
