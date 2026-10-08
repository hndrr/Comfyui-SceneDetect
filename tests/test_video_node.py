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
    def test_new_settings_preserve_existing_workflow_inputs_and_outputs(self):
        widget_names = [
            "method", "threshold", "min_scene_len_sec", "min_scene_len_frames",
            "luma_only", "representative", "max_width", "max_height",
            "limit_scenes", "write_thumbs", "thumbs_dir",
        ]
        for cls in (video_node.PySceneDetectVideo, legacy_node.PySceneDetectToImages):
            inputs = cls.INPUT_TYPES()
            names = list(inputs["required"]) + list(inputs["optional"])
            names = [name for name in names if name not in ("video", "image", "video_info")]
            self.assertEqual(names[:11], widget_names)
            self.assertEqual(names[11:15], [
                "hash_threshold", "hist_threshold", "downscale", "detector_settings",
            ])
            self.assertEqual(inputs["optional"]["detector_settings"][0], ["default", "custom"])
            self.assertEqual(inputs["optional"]["detector_settings"][1]["default"], "default")
            self.assertEqual(
                inputs["required"]["method"][0],
                ["content", "adaptive", "threshold", "hash", "histogram"],
            )
            self.assertEqual(cls.RETURN_TYPES[:3], ("IMAGE", "STRING", "INT"))
            self.assertEqual(cls.RETURN_NAMES[:3], ("images", "scenes_json", "scene_count"))

    def test_new_detectors_work_in_both_nodes_with_full_size_representatives(self):
        rng = np.random.default_rng(0)
        patterns = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
        hist_patterns = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        hist_patterns[0, :, 32:] = 255
        hist_patterns[1, :, 32:] = 128

        with tempfile.TemporaryDirectory() as tmpdir:
            for method, patterns, insensitive in (
                ("hash", patterns, {"hash_threshold": 1.0}),
                ("histogram", hist_patterns, {"hist_threshold": 0.9}),
            ):
                frames = np.repeat(patterns, 20, axis=0)
                video_path = Path(tmpdir) / f"{method}.avi"
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64)
                )
                self.assertTrue(writer.isOpened())
                try:
                    for frame in frames:
                        writer.write(frame)
                finally:
                    writer.release()

                for options, expected in (
                    ({}, [(0, 20), (20, 40)]),
                    (insensitive, [(0, 40)]),
                ):
                    settings = dict(
                        method=method, threshold=1000.0,
                        min_scene_len_sec=0.0, min_scene_len_frames=1,
                        luma_only=True, downscale=2, **options,
                    )
                    for kind in ("VIDEO", "Legacy VHS"):
                        with self.subTest(method=method, kind=kind, options=options):
                            if kind == "VIDEO":
                                result = video_node.PySceneDetectVideo().run(
                                    InputImpl.VideoFromFile(str(video_path)), **settings
                                )
                            else:
                                result = legacy_node.PySceneDetectToImages().run(
                                    torch.from_numpy(frames.copy()).float() / 255.0,
                                    {"loaded_fps": 10.0}, **settings,
                                )
                            images, scenes_json, count = result[:3]
                            scenes = json.loads(scenes_json)["scenes"]
                            self.assertEqual(count, len(expected))
                            self.assertEqual(images.shape, (count, 64, 64, 3))
                            self.assertEqual(
                                [(row["start_frame"], row["end_frame"]) for row in scenes],
                                expected,
                            )

    def test_custom_detector_settings_are_used_only_when_selected(self):
        adaptive_frames = np.repeat(
            np.array([np.zeros((32, 32, 3)), np.full((32, 32, 3), 255)], dtype=np.uint8),
            20, axis=0,
        )
        color_frames = np.zeros((40, 32, 32, 3), dtype=np.uint8)
        color_frames[:20, :, :, 0] = 255
        color_frames[20:, :, :, 2] = 255
        fade_frames = np.full((50, 32, 32, 3), 255, dtype=np.uint8)
        fade_frames[20:30] = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for method, frames, details, default_ranges, custom_ranges in (
                ("adaptive", adaptive_frames,
                 {"adaptive_threshold": 1000.0, "window_width": 1, "min_content_val": 0.0},
                 [(0, 20), (20, 40)], [(0, 40)]),
                ("content", color_frames,
                 {"delta_hue": 0.0, "delta_sat": 0.0, "delta_lum": 1.0, "delta_edges": 0.0},
                 [(0, 20), (20, 40)], [(0, 40)]),
                ("threshold", fade_frames, {"fade_bias": -1.0},
                 [(0, 25), (25, 50)], [(0, 20), (20, 50)]),
            ):
                video_path = Path(tmpdir) / f"custom-{method}.avi"
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32)
                )
                self.assertTrue(writer.isOpened())
                try:
                    for frame in frames:
                        writer.write(frame)
                finally:
                    writer.release()

                for mode, expected in (("default", default_ranges), ("custom", custom_ranges)):
                    settings = dict(
                        method=method, threshold=10.0,
                        min_scene_len_sec=0.0, min_scene_len_frames=1,
                        luma_only=False, detector_settings=mode, **details,
                    )
                    for kind in ("VIDEO", "Legacy VHS"):
                        with self.subTest(method=method, mode=mode, kind=kind):
                            if kind == "VIDEO":
                                result = video_node.PySceneDetectVideo().run(
                                    InputImpl.VideoFromFile(str(video_path)), **settings
                                )
                            else:
                                result = legacy_node.PySceneDetectToImages().run(
                                    torch.from_numpy(frames.copy()).float() / 255.0,
                                    {"loaded_fps": 10.0}, **settings,
                                )
                            images, scenes_json, count = result[:3]
                            self.assertEqual(count, len(expected))
                            self.assertEqual(images.shape, (count, 32, 32, 3))
                            self.assertEqual(
                                [(row["start_frame"], row["end_frame"])
                                 for row in json.loads(scenes_json)["scenes"]],
                                expected,
                            )

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

            result = video_node.PySceneDetectVideo().run(
                InputImpl.VideoFromFile(str(video_path)),
                method="content",
                threshold=10.0,
                min_scene_len_sec=0.0,
                min_scene_len_frames=1,
                luma_only=False,
            )
            images, scenes_json, count = result[:3]

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
                    result = video_node.PySceneDetectVideo().run(
                        InputImpl.VideoFromFile(
                            str(video_path), start_time=start_time, duration=duration
                        ),
                        method="content",
                        threshold=27.0,
                        min_scene_len_sec=0.0,
                        min_scene_len_frames=15,
                        luma_only=True,
                    )
                    images, scenes_json, count = result[:3]

                    self.assertEqual(count, 1)
                    self.assertEqual(images.shape, (1, 32, 32, 3))
                    self.assertAlmostEqual(images.mean().item(), 128 / 255, places=3)
                    scene = json.loads(scenes_json)["scenes"][0]
                    self.assertEqual(scene["index"], 1)
                    self.assertEqual(scene["start_frame"], start_frame)
                    self.assertEqual(scene["end_frame"], end_frame)
                    self.assertEqual(scene["duration_frames"], end_frame - start_frame)

    def test_legacy_video_without_cuts_returns_scene_and_frame(self):
        result = legacy_node.PySceneDetectToImages().run(
            torch.full((218, 32, 32, 3), 0.5),
            {"loaded_fps": 10.0},
            method="content",
            threshold=27.0,
            min_scene_len_sec=0.0,
            min_scene_len_frames=15,
            luma_only=True,
        )

        images, scenes_json, count = result[:3]
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
