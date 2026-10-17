import importlib
import builtins
import json
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import PropertyMock, patch

import torch
from scenedetect import FrameTimecode, SceneManager, open_video
from scenedetect.backends import AVAILABLE_BACKENDS
from scenedetect.common import Timecode

from test_video_node import InputImpl, legacy_node, video_node
from video_fixtures import write_timing_video


loader_node = (
    importlib.import_module(f"{video_node.__package__}.pyscenedetect_load_video")
    if video_node is not None else None
)
ops = (
    importlib.import_module(f"{video_node.__package__.rsplit('.', 1)[0]}.utils.video_ops")
    if video_node is not None else None
)


@unittest.skipIf(InputImpl is None, "ComfyUI's comfy_api is not available")
@unittest.skipUnless("pyav" in AVAILABLE_BACKENDS, "PyAV is not available")
class LegacyLoaderTests(unittest.TestCase):
    def load(self, path, **kwargs):
        with patch.object(loader_node.folder_paths, "get_annotated_filepath", return_value=str(path)):
            return loader_node.PySceneDetectLoadVideo().load_video(path.name, **kwargs)

    def test_loader_contract_and_example(self):
        cls = loader_node.PySceneDetectLoadVideo
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO"))
        self.assertEqual(cls.INPUT_TYPES()["required"]["backend"][1]["default"], "pyav")
        workflow_path = Path(__file__).resolve().parents[1] / "workflow/pyscene_workflow_legacy_pyav.json"
        workflow = json.loads(workflow_path.read_text())
        loader = next(n for n in workflow["nodes"] if n["type"] == "PySceneDetectLoadVideo")
        self.assertEqual(loader["widgets_values"][1:3], ["pyav", True])
        self.assertIn([20, 20, 3, 27, 1, "VHS_VIDEOINFO"], workflow["links"])
        self.assertIn([21, 20, 0, 27, 0, "IMAGE"], workflow["links"])

    def test_cfr_and_vfr_detection_match_upstream_for_both_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for vfr in (False, True):
                path = Path(tmpdir) / f"timing-{vfr}.mp4"
                write_timing_video(path, vfr=vfr)
                for backend in ("pyav", "opencv"):
                    images, count, audio, info = self.load(path, backend=backend, load_audio=False)
                    self.assertEqual(images.shape, (60, 64, 64, 3))
                    self.assertEqual(images.dtype, torch.float32)
                    self.assertEqual(count, 60)
                    self.assertIsNone(audio)
                    stream = open_video(str(path), backend=backend)
                    self.assertEqual(info["source_duration"], stream.duration.seconds)
                    expected_times = []
                    while stream.read() is not False:
                        expected_times.append(stream.position)
                    for actual, expected in zip(info["scenedetect_timecodes"], expected_times):
                        self.assertEqual((actual.pts, actual.time_base), (expected.pts, expected.time_base))
                    for method in ("content", "adaptive", "threshold", "hash", "histogram"):
                        manager = SceneManager()
                        manager.add_detector(ops.choose_detector(method, 10.0, 1, True))
                        stream = open_video(str(path), backend=backend)
                        manager.detect_scenes(video=stream)
                        expected = ops.timecodes_to_dict(manager.get_scene_list(start_in_scene=True), float(stream.frame_rate))
                        for representative in ("start", "middle", "end"):
                            with self.subTest(vfr=vfr, backend=backend, method=method, representative=representative):
                                with (
                                    patch("av.open", side_effect=AssertionError("Legacy reopened the source")),
                                    patch("cv2.VideoCapture", side_effect=AssertionError("Legacy reopened the source")),
                                ):
                                    reps, payload, scenes = legacy_node.PySceneDetectToImages().run(
                                        images, info, method, 10.0, 0.0, 1, True,
                                        representative=representative,
                                    )
                                result = json.loads(payload)
                                self.assertEqual(result["scenes"], expected)
                                self.assertEqual(scenes, len(expected))
                                self.assertNotIn("scenedetect_timecodes", result["video_info"])
                                if method == "content":
                                    self.assertEqual(scenes, 2)
                                    self.assertLess(reps[0].mean().item(), 0.02)
                                    self.assertGreater(reps[1].mean().item(), 0.98)
                                    self.assertEqual(result["scenes"][1]["start_time"], "00:00:03.000" if vfr else "00:00:01.000")

    def test_loader_does_not_override_upstream_backend_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fallback.mp4"
            write_timing_video(path)
            with patch.dict(AVAILABLE_BACKENDS, {"opencv": AVAILABLE_BACKENDS["opencv"]}, clear=True):
                images, count, _, info = self.load(path, backend="pyav", load_audio=False)
            self.assertEqual(count, 60)
            self.assertEqual(len(info["scenedetect_timecodes"]), len(images))

    def test_legacy_uses_modified_images_and_rejects_stale_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vfr.mp4"
            write_timing_video(path, vfr=True)
            images, _, _, info = self.load(path, load_audio=False)
        images = images * 0.5 + 0.25
        reps, payload, count = legacy_node.PySceneDetectToImages().run(
            images, info, "content", 10.0, 0.0, 1, True, representative="middle",
        )
        self.assertEqual(count, 2)
        self.assertAlmostEqual(reps[0].mean().item(), 63 / 255, places=3)
        self.assertAlmostEqual(reps[1].mean().item(), 191 / 255, places=3)
        self.assertEqual(json.loads(payload)["scenes"][1]["start_time"], "00:00:03.000")
        with self.assertRaisesRegex(ValueError, "same frame count"):
            legacy_node.PySceneDetectToImages().run(images[:30], info, "content", 10.0, 0.0, 1, True)
        with self.assertRaisesRegex(ValueError, "presentation order"):
            ops.TensorVideoStream(images, 30.0, tuple(reversed(info["scenedetect_timecodes"])))

    def test_timestamp_stream_retains_timecodes_and_seeks_batch_indices(self):
        times = tuple(FrameTimecode(Timecode(t, Fraction(1, 1000)), 30.0) for t in (0, 100, 101, 200))
        stream = ops.TensorVideoStream(torch.zeros((4, 8, 8, 3)), 30.0, times)
        stream.seek(times[2])
        self.assertEqual(stream.frame_number, 2)
        stream.read()
        self.assertIs(stream.position, times[2])
        self.assertEqual(stream.duration, times[-1] + 1)
        stream.reset()
        self.assertEqual(stream.position.frame_num, 0)
        for target, expected in ((2, 2), (times[2], 2), (0.1, 1)):
            with self.subTest(target=target):
                stream.seek(target)
                self.assertEqual(stream.frame_number, expected)
                stream.read()
                self.assertIs(stream.position, times[expected])
        with self.assertRaises(ValueError):
            stream.seek(-1)

    def test_missing_audio_dependencies_do_not_break_video_or_upstream_fallback(self):
        original_import = builtins.__import__
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.mp4"
            write_timing_video(path, audio=True)
            for missing in ("av", "comfy_extras.nodes_audio"):
                def import_without_audio(name, *args, **kwargs):
                    if name == missing:
                        raise ImportError(f"Missing {missing}")
                    return original_import(name, *args, **kwargs)

                for backend in ("opencv", "pyav"):
                    with self.subTest(missing=missing, backend=backend):
                        with (
                            patch.dict(AVAILABLE_BACKENDS, {"opencv": AVAILABLE_BACKENDS["opencv"]}, clear=True),
                            patch("builtins.__import__", side_effect=import_without_audio),
                            self.assertLogs(level="WARNING") as logs,
                        ):
                            images, count, audio, info = self.load(path, backend=backend)
                        self.assertEqual(count, 60)
                        self.assertEqual(len(images), len(info["scenedetect_timecodes"]))
                        self.assertIsNone(audio)
                        self.assertIn("audio", " ".join(logs.output).lower())

    def test_loader_does_not_concatenate_float32_frame_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.mp4"
            write_timing_video(path)
            with patch.object(loader_node.torch, "cat", side_effect=AssertionError("Duplicate float32 batch")):
                images, count, _, _ = self.load(path, load_audio=False)
            self.assertEqual(images.shape, (60, 64, 64, 3))
            self.assertEqual(count, 60)
            self.assertEqual(images.dtype, torch.float32)
            self.assertLess(images[0].mean().item(), 0.02)
            self.assertGreater(images[-1].mean().item(), 0.98)

    def test_source_estimates_are_not_replaced_with_decoded_counts(self):
        images = torch.zeros((2, 8, 8, 3))
        images[0, :, :, 0] = 1.0
        stream = ops.TensorVideoStream(images, 30.0)
        # The source estimate and the number actually decoded have different contracts.
        duration = FrameTimecode(1234, 30.0)
        with (
            patch.object(loader_node, "open_video", return_value=stream),
            patch.object(type(stream), "duration", new_callable=PropertyMock, return_value=duration),
        ):
            loaded, count, _, info = self.load(Path("estimate.mp4"), load_audio=False)
        self.assertEqual(count, 2)
        self.assertEqual(info["loaded_frame_count"], 2)
        self.assertEqual(info["source_frame_count"], 1234)
        self.assertEqual(info["source_duration"], duration.seconds)
        torch.testing.assert_close(loaded, images)

    def test_loader_stops_decoding_on_comfyui_cancellation(self):
        stream = ops.TensorVideoStream(torch.zeros((60, 8, 8, 3)), 30.0)
        with (
            patch.object(loader_node, "open_video", return_value=stream),
            patch.object(
                loader_node.model_management, "throw_exception_if_processing_interrupted",
                side_effect=[None, None, RuntimeError("cancelled")],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.load(Path("cancel.mp4"), load_audio=False)
        self.assertEqual(stream.frame_number, 2)

    def test_short_vfr_scenes_keep_legacy_representatives_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "short-vfr.mp4"
            write_timing_video(path, vfr=True, short_scenes=True)
            for backend in ("opencv", "pyav"):
                images, _, _, info = self.load(path, backend=backend, load_audio=False)
                for representative in ("start", "middle", "end"):
                    with self.subTest(backend=backend, representative=representative):
                        reps, payload, count = legacy_node.PySceneDetectToImages().run(
                            images, info, "content", 10.0, 0.0, 1, True,
                            representative=representative,
                        )
                        self.assertEqual(count, 6)
                        self.assertEqual(len(reps), count)
                        self.assertEqual(len(json.loads(payload)["scenes"]), count)
                        for image, expected in zip(reps, (0, 255, 128, 30, 200, 70)):
                            self.assertAlmostEqual(image.mean().item() * 255, expected, delta=3)

    def test_audio_and_silent_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.mp4"
            write_timing_video(path, audio=True)
            # The core audio decoder does not use the server; avoid starting core nodes
            # when importing it from a standalone custom-node test process.
            server = types.ModuleType("server")
            server.PromptServer = object
            with patch.dict(sys.modules, {"server": server}):
                _, _, audio, _ = self.load(path)
            self.assertEqual(audio["sample_rate"], 48000)
            self.assertEqual(audio["waveform"].shape[:2], (1, 2))
            self.assertGreater(audio["waveform"].shape[-1], 95000)
            self.assertEqual(audio["waveform"].dtype, torch.float32)
            self.assertIsNone(self.load(path, load_audio=False)[2])
            write_timing_video(path)
            self.assertIsNone(self.load(path)[2])

    def test_input_path_is_resolved_before_opening(self):
        with (
            patch.object(loader_node.folder_paths, "get_annotated_filepath", side_effect=ValueError("Invalid file path")),
            patch.object(loader_node, "open_video") as open_stream,
        ):
            with self.assertRaisesRegex(ValueError, "Invalid file path"):
                loader_node.PySceneDetectLoadVideo().load_video("../outside.mp4")
            open_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
