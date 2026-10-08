import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scenedetect import FrameTimecode, is_ffmpeg_available

from utils.scene_clips import FileBackedVideo, load_video_from_file, sanitize_clip_name, split_scene_clips
from test_video_node import InputImpl, video_node


class SceneClipTests(unittest.TestCase):
    def test_load_and_save_without_comfy_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "clip.mp4"
            source.write_bytes(b"clip")
            with patch.dict("sys.modules", {"comfy_api.latest": None}):
                clip = load_video_from_file(str(source))
            self.assertIsInstance(clip, FileBackedVideo)
            self.assertEqual(clip.get_stream_source(), str(source))
            dest = Path(tmpdir) / "nested" / "copy.mp4"
            clip.save_to(dest)
            self.assertEqual(dest.read_bytes(), b"clip")
            clip.save_to(source)
            self.assertEqual(source.read_bytes(), b"clip")

    def test_names_and_missing_ffmpeg(self):
        self.assertEqual(sanitize_clip_name("My Video (1).mov"), "My_Video_1")
        self.assertEqual(sanitize_clip_name(""), "scene")
        self.assertEqual(split_scene_clips("unused", [], "unused"), [])
        scenes = [(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))]
        with patch("utils.scene_clips.is_ffmpeg_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg is required"):
                split_scene_clips("unused", scenes, "unused")

    def test_stream_copy_failure_retries_with_reencode(self):
        scenes = [(FrameTimecode(0, 10.0), FrameTimecode(20, 10.0))]
        attempts = []
        with tempfile.TemporaryDirectory() as tmpdir:
            def split(*args, **kwargs):
                attempts.append(kwargs["arg_override"])
                if len(attempts) == 1:
                    return 1
                (Path(tmpdir) / "safe-Scene-001.mp4").write_bytes(b"clip")
                return 0
            with patch("utils.scene_clips.is_ffmpeg_available", return_value=True), patch("utils.scene_clips.split_video_ffmpeg", side_effect=split):
                clips = split_scene_clips("input.mp4", scenes, tmpdir, video_name="safe", reencode=False)
            self.assertEqual(clips, [str(Path(tmpdir) / "safe-Scene-001.mp4")])
            self.assertIn("-c copy", attempts[0])
            self.assertIn("libx264", attempts[1])

    @unittest.skipIf(InputImpl is None, "ComfyUI's comfy_api is not available")
    @unittest.skipUnless(is_ffmpeg_available(), "ffmpeg is required")
    def test_clips_follow_scene_limit_and_trim_and_stay_in_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hard-cut.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for value in (0, 255):
                    for _ in range(20):
                        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
            finally:
                writer.release()
            output = Path(tmpdir) / "output"
            temp = Path(tmpdir) / "temp"
            output.mkdir()
            temp.mkdir()
            with patch.object(video_node.folder_paths, "get_output_directory", return_value=str(output)), patch.object(video_node.folder_paths, "get_temp_directory", return_value=str(temp)):
                for trim_start, duration, limit, expected_ranges in (
                    (0.0, 0.0, 0, [(0, 20), (20, 40)]),
                    (1.0, 2.0, 1, [(10, 20)]),
                ):
                    with self.subTest(trim=trim_start, limit=limit):
                        result = video_node.PySceneDetectVideo().run(
                            InputImpl.VideoFromFile(str(path), start_time=trim_start, duration=duration),
                            method="content", threshold=10.0, min_scene_len_sec=0.0,
                            min_scene_len_frames=1, luma_only=False, limit_scenes=limit,
                            split_clips=True,
                        )
                        images, scenes_json, count, clips = result
                        scenes = json.loads(scenes_json)["scenes"]
                        self.assertEqual(count, len(expected_ranges))
                        self.assertEqual(images.shape, (count, 32, 32, 3))
                        self.assertEqual([(row["start_frame"], row["end_frame"]) for row in scenes], expected_ranges)
                        self.assertEqual(len(clips), count)
                        for scene, clip in zip(scenes, clips):
                            clip_path = Path(clip.get_stream_source())
                            self.assertTrue(clip_path.is_file())
                            self.assertTrue(clip_path.is_relative_to(temp))
                            self.assertEqual(str(clip_path), scene["clip_path"])
                            self.assertAlmostEqual(clip.scene_duration_sec, scene["duration_sec"])
                            self.assertEqual(clip.get_frame_count(), scene["duration_frames"])
                with patch.object(video_node, "split_scene_clips", side_effect=AssertionError("splitting is off by default")):
                    result = video_node.PySceneDetectVideo().run(
                        InputImpl.VideoFromFile(str(path)), method="content", threshold=10.0,
                        min_scene_len_sec=0.0, min_scene_len_frames=1, luma_only=False,
                    )
                    self.assertEqual(result[3], [])
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
