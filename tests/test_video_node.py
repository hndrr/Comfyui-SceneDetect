import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scenedetect import ContentDetector, detect
from scenedetect.backends import AVAILABLE_BACKENDS

from video_fixtures import write_timing_video

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
    def test_backend_is_appended_and_example_keeps_existing_widget_values(self):
        inputs = video_node.PySceneDetectVideo.INPUT_TYPES()
        self.assertEqual(list(inputs["optional"])[-1], "backend")
        self.assertEqual(inputs["optional"]["backend"], (["opencv", "pyav"], {"default": "opencv"}))
        self.assertNotIn("backend", legacy_node.PySceneDetectToImages.INPUT_TYPES()["optional"])

        workflow_path = Path(__file__).resolve().parents[1] / "workflow" / "pyscene_workflow.json"
        workflow = json.loads(workflow_path.read_text())
        node = next(node for node in workflow["nodes"] if node["type"] == "PySceneDetectVideo")
        widget_names = list(inputs["required"])[1:] + list(inputs["optional"])
        legacy_inputs = legacy_node.PySceneDetectToImages.INPUT_TYPES()
        legacy_widget_names = [
            name for group in ("required", "optional") for name in legacy_inputs[group]
            if name not in ("image", "video_info")
        ]
        self.assertEqual(widget_names[:-1], legacy_widget_names)
        self.assertEqual(len(node["widgets_values"]), len(widget_names))
        self.assertEqual(node["widgets_values"][-1], "opencv")
        self.assertEqual(node["widgets_values"][:11], [
            "content", 27, 0, 1, True, "start", 0, 0, 0, False, "",
        ])

    @unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
    def test_backends_keep_upstream_vfr_metadata_and_correct_middle_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vfr.mp4"
            write_timing_video(path, vfr=True)
            for backend in ("opencv", "pyav"):
                with self.subTest(backend=backend):
                    upstream = detect(
                        str(path), ContentDetector(threshold=10.0, min_scene_len=1, luma_only=True),
                        start_in_scene=True, backend=backend,
                    )
                    images, scenes_json, count = video_node.PySceneDetectVideo().run(
                        InputImpl.VideoFromFile(str(path)), "content", 10.0, 0.0, 1, True,
                        representative="middle", backend=backend,
                    )
                    metadata = json.loads(scenes_json)
                    self.assertEqual(metadata["backend"], backend)
                    self.assertEqual(count, 2)
                    self.assertEqual(images.shape, (2, 64, 64, 3))
                    self.assertLess(images[0].mean().item(), 0.02)
                    self.assertGreater(images[1].mean().item(), 0.98)
                    for row, (start, end) in zip(metadata["scenes"], upstream):
                        self.assertEqual(row["start_frame"], start.frame_num)
                        self.assertEqual(row["end_frame"], end.frame_num)
                        self.assertEqual(row["start_time"], start.get_timecode())
                        self.assertEqual(row["end_time"], end.get_timecode())
                        self.assertEqual(row["duration_sec"], end.seconds - start.seconds)

    @unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
    def test_backends_support_representatives_trim_limits_resize_and_thumbnails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfr.mp4"
            write_timing_video(path)
            for backend in ("opencv", "pyav"):
                for representative in ("start", "middle", "end"):
                    with self.subTest(backend=backend, representative=representative):
                        images, _, count = video_node.PySceneDetectVideo().run(
                            InputImpl.VideoFromFile(str(path), start_time=0.5, duration=1.0),
                            "content", 10.0, 0.0, 1, True,
                            representative=representative, backend=backend,
                        )
                        self.assertEqual(count, 2)
                        self.assertEqual(images.shape, (2, 64, 64, 3))
                        self.assertLess(images[0].mean().item(), 0.02)
                        self.assertGreater(images[1].mean().item(), 0.98)

                with patch.object(video_node.folder_paths, "get_output_directory", return_value=tmpdir):
                    images, metadata, count = video_node.PySceneDetectVideo().run(
                        InputImpl.VideoFromFile(str(path)), "content", 10.0, 0.0, 1, True,
                        backend=backend, limit_scenes=1, max_width=32,
                        write_thumbs=True, thumbs_dir=backend,
                    )
                self.assertEqual(count, 1)
                self.assertEqual(images.shape, (1, 32, 32, 3))
                self.assertEqual(len(json.loads(metadata)["scenes"]), 1)
                thumbnail = cv2.imread(str(Path(tmpdir) / backend / "scene_001_f0.jpg"))
                self.assertEqual(thumbnail.shape, (32, 32, 3))

    @unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
    def test_vfr_short_scenes_keep_all_representatives_inside_scene(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "short-vfr.mp4"
            write_timing_video(path, vfr=True, short_scenes=True)
            for backend in ("opencv", "pyav"):
                upstream = detect(
                    str(path), ContentDetector(threshold=10.0, min_scene_len=1, luma_only=True),
                    start_in_scene=True, backend=backend,
                )
                for representative in ("start", "middle", "end"):
                    with self.subTest(backend=backend, representative=representative):
                        with patch.object(video_node.folder_paths, "get_output_directory", return_value=tmpdir):
                            images, payload, count = video_node.PySceneDetectVideo().run(
                                InputImpl.VideoFromFile(str(path)), "content", 10.0, 0.0, 1, True,
                                representative=representative, backend=backend,
                                write_thumbs=True, thumbs_dir=f"{backend}-{representative}",
                            )
                        self.assertEqual(count, len(upstream))
                        self.assertEqual(len(images), count)
                        self.assertEqual(count, 6)
                        for image, expected in zip(images, (0, 255, 128, 30, 200, 70)):
                            self.assertAlmostEqual(image.mean().item() * 255, expected, delta=3)
                        for row, (start, end) in zip(json.loads(payload)["scenes"], upstream):
                            self.assertEqual(row["start_time"], start.get_timecode())
                            self.assertEqual(row["end_time"], end.get_timecode())
                        thumbs = list((Path(tmpdir) / f"{backend}-{representative}").glob("*.jpg"))
                        self.assertEqual(len(thumbs), count)

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
                cv2.VideoWriter_fourcc(*"FFV1"),  # Preserve exact pixel values across codec builds.
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
