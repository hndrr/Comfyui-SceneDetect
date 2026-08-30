import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scenedetect import is_ffmpeg_available


def _install_folder_paths_stub():
    if "folder_paths" in sys.modules:
        return sys.modules["folder_paths"]

    module = types.ModuleType("folder_paths")

    def is_within_directory(root: str, path: str) -> bool:
        root = os.path.realpath(root)
        path = os.path.realpath(path)
        try:
            return os.path.commonpath([root, path]) == root
        except ValueError:
            return False

    module.get_output_directory = lambda: os.getcwd()
    module.get_temp_directory = lambda: os.getcwd()
    module.is_within_directory = is_within_directory
    sys.modules["folder_paths"] = module
    return module


def _load_video_node():
    _install_folder_paths_stub()
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


class FakeVideo:
    def __init__(self, path: Path):
        self._path = str(path)
        capture = cv2.VideoCapture(self._path)
        try:
            self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)
            self._count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()

    def get_frame_rate(self) -> float:
        return self._fps

    def get_dimensions(self):
        return self._width, self._height

    def get_frame_count(self) -> int:
        return self._count

    def get_duration(self) -> float:
        return self._count / self._fps if self._fps else 0.0

    def get_active_trim_window(self):
        return 0.0, 0.0

    def get_stream_source(self) -> str:
        return self._path


def _write_hard_cut(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 32),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    try:
        for value in (0, 255):
            frame = np.full((32, 32, 3), value, dtype=np.uint8)
            for _ in range(20):
                writer.write(frame)
    finally:
        writer.release()


video_node = _load_video_node()


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

    def test_official_video_input_finds_scenes_and_llm_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            _write_hard_cut(video_path)
            images, scenes_json, count, all_scenes_text, per_scene_prompt_list, videos = (
                video_node.PySceneDetectVideo().run(
                    FakeVideo(video_path),
                    method="content",
                    threshold=10.0,
                    min_scene_len_sec=0.0,
                    min_scene_len_frames=1,
                    luma_only=False,
                    prompt_template="Scene {index}/{scene_count}",
                )
            )

        scenes = json.loads(scenes_json)["scenes"]
        self.assertEqual(count, 2)
        self.assertEqual(images.shape, (2, 32, 32, 3))
        self.assertEqual(
            [(scene["start_frame"], scene["end_frame"]) for scene in scenes],
            [(0, 20), (20, 40)],
        )
        self.assertIn("# Scenes (2)", all_scenes_text)
        self.assertEqual(per_scene_prompt_list, ["Scene 1/2", "Scene 2/2"])
        self.assertEqual(videos, [])

    def test_v1_input_types_omit_show_all_settings(self):
        types = video_node.PySceneDetectVideo.INPUT_TYPES()
        self.assertNotIn("show_all_settings", types["required"])
        self.assertNotIn("show_all_settings", types["optional"])
        self.assertEqual(
            types["required"]["method"][0],
            ["content", "adaptive", "threshold", "hash", "histogram"],
        )

    def test_run_accepts_dynamic_combo_method_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "hard-cut.avi"
            _write_hard_cut(video_path)
            images, scenes_json, count, _, _, videos = (
                video_node.PySceneDetectVideo().run(
                    FakeVideo(video_path),
                    method={
                        "method": "content",
                        "threshold": 10.0,
                        "luma_only": False,
                    },
                    threshold=999.0,
                    min_scene_len_sec=0.0,
                    min_scene_len_frames=1,
                    luma_only=True,
                )
            )

        scenes = json.loads(scenes_json)["scenes"]
        self.assertEqual(count, 2)
        self.assertEqual(images.shape, (2, 32, 32, 3))
        self.assertEqual(
            [(scene["start_frame"], scene["end_frame"]) for scene in scenes],
            [(0, 20), (20, 40)],
        )
        self.assertEqual(json.loads(scenes_json)["threshold"], 10.0)
        self.assertEqual(videos, [])

    @unittest.skipUnless(is_ffmpeg_available(), "ffmpeg is required to split clips")
    def test_split_clips_keeps_files_in_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            output_root.mkdir()
            temp_root.mkdir()
            video_path = Path(tmpdir) / "hard-cut.avi"
            _write_hard_cut(video_path)
            folder_paths = sys.modules["folder_paths"]
            folder_paths.get_output_directory = lambda: str(output_root)
            folder_paths.get_temp_directory = lambda: str(temp_root)

            with patch.object(video_node, "load_video_from_file", side_effect=lambda path: path):
                (
                    _images,
                    scenes_json,
                    count,
                    _all_scenes_text,
                    _per_scene_prompt_list,
                    videos,
                ) = video_node.PySceneDetectVideo().run(
                    FakeVideo(video_path),
                    method="content",
                    threshold=10.0,
                    min_scene_len_sec=0.0,
                    min_scene_len_frames=1,
                    luma_only=False,
                    split_clips=True,
                    split_reencode=True,
                )

            scenes = json.loads(scenes_json)["scenes"]
            self.assertEqual(count, 2)
            self.assertEqual(len(videos), 2)
            for scene, clip in zip(scenes, videos):
                clip_path = Path(clip)
                self.assertTrue(clip_path.is_file())
                self.assertEqual(scene["clip_path"], clip)
                self.assertTrue(clip_path.resolve().is_relative_to(temp_root.resolve()))
                self.assertFalse(
                    clip_path.resolve().is_relative_to(output_root.resolve())
                )


if __name__ == "__main__":
    unittest.main()
