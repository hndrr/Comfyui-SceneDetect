import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_folder_paths_stub(output_root: str, temp_root: str):
    module = sys.modules.get("folder_paths")
    if module is None:
        module = types.ModuleType("folder_paths")
        sys.modules["folder_paths"] = module

    def is_within_directory(root: str, path: str) -> bool:
        root = os.path.realpath(root)
        path = os.path.realpath(path)
        try:
            return os.path.commonpath([root, path]) == root
        except ValueError:
            return False

    module.get_output_directory = lambda: output_root
    module.get_temp_directory = lambda: temp_root
    module.is_within_directory = is_within_directory
    return module


def _load_preview_node(output_root: str, temp_root: str):
    _install_folder_paths_stub(output_root, temp_root)
    repository_root = Path(__file__).resolve().parents[1]
    package_name = "comfyui_scenedetect_preview_test_package"

    package = types.ModuleType(package_name)
    package.__path__ = [str(repository_root)]
    sys.modules[package_name] = package

    nodes_package_name = f"{package_name}.nodes"
    nodes_package = types.ModuleType(nodes_package_name)
    nodes_package.__path__ = [str(repository_root / "nodes")]
    sys.modules[nodes_package_name] = nodes_package

    sys.modules.pop(f"{nodes_package_name}.pyscenedetect_preview", None)
    return importlib.import_module(f"{nodes_package_name}.pyscenedetect_preview")


def _write_dummy_video(path: Path) -> None:
    path.write_bytes(b"fake-video")


class FakeVideo:
    def __init__(self, path: Path | None, save_to_impl=None, duration_sec=None):
        self._path = str(path) if path is not None else None
        self._save_to_impl = save_to_impl
        if duration_sec is not None:
            self.scene_duration_sec = duration_sec

    def get_stream_source(self):
        return self._path

    def save_to(self, dest_path: str):
        if self._save_to_impl is not None:
            self._save_to_impl(dest_path)
            return
        raise AssertionError("save_to should not be called when a file path exists")


class PreviewNodeTests(unittest.TestCase):
    def test_temp_path_is_returned_without_copying_or_writing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            clip_dir = temp_root / "scenedetect_clips" / "abc"
            output_root.mkdir()
            clip_dir.mkdir(parents=True)
            clip = clip_dir / "scene_001.mp4"
            _write_dummy_video(clip)

            preview_node = _load_preview_node(str(output_root), str(temp_root))
            result = preview_node.preview_entry_for_path(str(clip))

            self.assertEqual(result["filename"], "scene_001.mp4")
            self.assertEqual(result["subfolder"], "scenedetect_clips/abc")
            self.assertEqual(result["type"], "temp")
            self.assertEqual(list(output_root.rglob("*")), [])
            self.assertEqual(clip.read_bytes(), b"fake-video")

    def test_outside_path_is_copied_into_temp_preview_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            source_dir = Path(tmpdir) / "source"
            output_root.mkdir()
            temp_root.mkdir()
            source_dir.mkdir()
            source = source_dir / "clip.mp4"
            _write_dummy_video(source)

            preview_node = _load_preview_node(str(output_root), str(temp_root))
            result = preview_node.preview_entry_for_path(str(source))

            copied = temp_root / result["subfolder"] / result["filename"]
            self.assertEqual(result["subfolder"], "scenedetect_preview")
            self.assertEqual(result["type"], "temp")
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), b"fake-video")
            self.assertTrue(source.is_file())
            self.assertEqual(list(output_root.rglob("*")), [])

    def test_preview_node_renders_list_of_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            clip_dir = temp_root / "scenedetect_clips"
            output_root.mkdir()
            clip_dir.mkdir(parents=True)
            clips = [clip_dir / "scene_001.mp4", clip_dir / "scene_002.mp4"]
            for clip in clips:
                _write_dummy_video(clip)

            preview_node = _load_preview_node(str(output_root), str(temp_root))
            payload = preview_node.PySceneDetectPreviewVideos().preview(
                [FakeVideo(clip) for clip in clips]
            )

            videos = payload["ui"]["scene_previews"]
            self.assertEqual(len(videos), 2)
            self.assertNotIn("images", payload["ui"])
            self.assertNotIn("videos", payload["ui"])
            self.assertEqual(
                [entry["filename"] for entry in videos],
                ["scene_001.mp4", "scene_002.mp4"],
            )
            self.assertEqual(list(output_root.rglob("*")), [])

    def test_preview_includes_detected_scene_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            clip_dir = temp_root / "scenedetect_clips"
            output_root.mkdir()
            clip_dir.mkdir(parents=True)
            clip = clip_dir / "scene_001.mp4"
            _write_dummy_video(clip)

            preview_node = _load_preview_node(str(output_root), str(temp_root))
            payload = preview_node.PySceneDetectPreviewVideos().preview(
                FakeVideo(clip, duration_sec=1.25)
            )

            self.assertEqual(payload["ui"]["scene_previews"][0]["duration_sec"], 1.25)

    def test_save_to_fallback_writes_only_to_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            output_root.mkdir()
            temp_root.mkdir()

            def save_to(dest_path: str):
                Path(dest_path).write_bytes(b"from-save-to")

            preview_node = _load_preview_node(str(output_root), str(temp_root))
            payload = preview_node.PySceneDetectPreviewVideos().preview(
                FakeVideo(None, save_to_impl=save_to)
            )

            entry = payload["ui"]["scene_previews"][0]
            copied = temp_root / entry["subfolder"] / entry["filename"]
            self.assertEqual(entry["subfolder"], "scenedetect_preview")
            self.assertEqual(entry["type"], "temp")
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_bytes(), b"from-save-to")
            self.assertEqual(list(output_root.rglob("*")), [])

    def test_empty_list_returns_empty_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "output"
            temp_root = Path(tmpdir) / "temp"
            output_root.mkdir()
            temp_root.mkdir()
            preview_node = _load_preview_node(str(output_root), str(temp_root))
            payload = preview_node.PySceneDetectPreviewVideos().preview([])
            self.assertEqual(payload["ui"]["scene_previews"], [])
            self.assertNotIn("images", payload["ui"])
            self.assertNotIn("videos", payload["ui"])


if __name__ == "__main__":
    unittest.main()
