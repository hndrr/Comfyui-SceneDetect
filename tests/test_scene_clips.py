import io
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import torch

from test_preview_node import _load_preview_node
from test_video_node import InputImpl, video_node
from video_fixtures import write_timing_video


@unittest.skipIf(InputImpl is None, "ComfyUI's comfy_api is not available")
class SceneClipTests(unittest.TestCase):
    def test_scene_videos_are_lazy_and_respect_trim_and_scene_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.mp4"
            write_timing_video(source)
            original = source.read_bytes()
            output, temp = root / "output", root / "temp"
            output.mkdir()
            temp.mkdir()
            with patch.object(video_node.folder_paths, "get_output_directory", return_value=str(output)), \
                 patch.object(video_node.folder_paths, "get_temp_directory", return_value=str(temp)), \
                 patch.object(InputImpl.VideoFromFile, "save_to", side_effect=AssertionError("Scene detection must not encode clips")):
                for start, duration, limit, expected_windows in (
                    (0.0, 0.0, 0, [(0.0, 1.0), (1.0, 1.0)]),
                    (0.5, 1.0, 0, [(0.5, 0.5), (1.0, 0.5)]),
                    (0.5, 1.0, 1, [(0.5, 0.5)]),
                    (-0.5, 0.0, 0, [(1.5, 0.5)]),
                ):
                    with self.subTest(start=start, duration=duration, limit=limit):
                        images, metadata, count, clips = video_node.PySceneDetectVideo().run(
                            InputImpl.VideoFromFile(str(source), start_time=start, duration=duration),
                            "content", 10.0, 0.0, 1, True, limit_scenes=limit,
                        )
                        self.assertEqual(count, len(expected_windows))
                        self.assertEqual(images.shape[0], count)
                        self.assertEqual(len(clips), count)
                        self.assertEqual([clip.get_active_trim_window() for clip in clips], expected_windows)
                        for row, clip in zip(json.loads(metadata)["scenes"], clips):
                            self.assertIsInstance(clip, InputImpl.VideoFromFile)
                            self.assertEqual(clip.get_stream_source(), str(source))
                            self.assertAlmostEqual(clip.get_duration(), row["duration_sec"])
                            self.assertNotIn("clip_path", row)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(temp.iterdir()), [])
            self.assertEqual(source.read_bytes(), original)

    def test_buffer_backed_clips_outlive_the_detection_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.mp4"
            write_timing_video(source)
            buffer = io.BytesIO(source.read_bytes())
            with patch.object(video_node, "detect_scenes", wraps=video_node.detect_scenes) as detect:
                _, _, count, clips = video_node.PySceneDetectVideo().run(
                    InputImpl.VideoFromFile(buffer), "content", 10.0, 0.0, 1, True,
                )
            self.assertEqual(count, 2)
            self.assertFalse(Path(detect.call_args.args[0]).exists())
            for index, clip in enumerate(clips):
                self.assertIs(clip.get_stream_source(), buffer)
                saved = root / f"clip-{index}.mp4"
                clip.save_to(str(saved))
                with av.open(str(saved)) as container:
                    frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 30)
                self.assertAlmostEqual(frames[0].to_ndarray(format="rgb24").mean(), 255 * index, delta=3)

    def test_component_video_source_is_encoded_once_for_all_scenes(self):
        from comfy_api.latest import Types

        frames = torch.zeros((60, 32, 32, 3))
        frames[30:] = 1.0
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, frame_rate=Fraction(30)))
        with patch.object(video, "save_to", wraps=video.save_to) as encode, \
             patch.object(video, "get_stream_source", wraps=video.get_stream_source) as get_source:
            _, _, count, clips = video_node.PySceneDetectVideo().run(video, "content", 10.0, 0.0, 1, True)
        self.assertEqual(count, 2)
        self.assertEqual(get_source.call_count, 1)
        self.assertEqual(encode.call_count, 1)
        self.assertIsInstance(clips[0].get_stream_source(), io.BytesIO)
        self.assertIs(clips[0].get_stream_source(), clips[1].get_stream_source())

    def test_saved_and_previewed_cfr_vfr_and_short_scenes_match_source_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output, temp = root / "output", root / "temp"
            output.mkdir()
            temp.mkdir()
            preview = _load_preview_node(str(output), str(temp))
            for vfr in (False, True):
                source = root / f"source-{vfr}.mp4"
                write_timing_video(source, vfr=vfr, short_scenes=True)
                original = source.read_bytes()
                with av.open(str(source)) as container:
                    source_frames = [(float(frame.pts * frame.time_base), frame.to_ndarray(format="rgb24").mean())
                                     for frame in container.decode(video=0)]
                for backend in ("opencv", "pyav"):
                    _, _, count, clips = video_node.PySceneDetectVideo().run(
                        InputImpl.VideoFromFile(str(source)), "content", 10.0, 0.0, 1, True, backend=backend,
                    )
                    self.assertEqual(len(clips), count)
                    for index, clip in enumerate(clips):
                        with self.subTest(vfr=vfr, backend=backend, scene=index):
                            start, duration = clip.get_active_trim_window()
                            # OpenCV rounds source timestamps to microseconds.
                            expected = [mean for pts, mean in source_frames if start - 1e-6 <= pts < start + duration - 1e-6]
                            saved = root / f"saved-{vfr}-{backend}-{index}.mp4"
                            clip.save_to(str(saved))
                            entry = preview.preview_entry_for_video(clip)
                            preview_path = temp / entry["subfolder"] / entry["filename"]
                            self.assertEqual(entry["type"], "temp")
                            for path in (saved, preview_path):
                                with av.open(str(path)) as container:
                                    actual = list(container.decode(video=0))
                                self.assertEqual(len(actual), len(expected))
                                self.assertAlmostEqual(float(actual[0].pts * actual[0].time_base), 0.0, delta=1e-3)
                                for frame, mean in zip(actual, expected):
                                    self.assertAlmostEqual(frame.to_ndarray(format="rgb24").mean(), mean, delta=3)
                self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
