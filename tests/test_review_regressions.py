import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import imageio_ffmpeg
import torch

from test_preview_node import _load_preview_node
from test_scene_text import text_node
from test_video_node import InputImpl, legacy_node, video_node
from video_fixtures import write_timing_video


class ReviewRegressionTests(unittest.TestCase):
    def test_video_output_is_available_by_default_and_empty_scenes_skip_consumers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / 'source.mp4'
            write_timing_video(source)
            node = video_node.PySceneDetectVideo()
            result = node.run(InputImpl.VideoFromFile(str(source)), 'content', 10.0, 0.0, 1, True)
            self.assertEqual(result[0].shape[0], 2)
            self.assertEqual(result[2], 2)
            self.assertEqual(len(text_node.PySceneDetectSceneText.execute(result[1]).result[1]), 2)
            self.assertEqual(len(result[3]), 2)
            self.assertTrue(all(isinstance(clip, InputImpl.VideoFromFile) for clip in result[3]))
            with patch.object(video_node, 'detect_scenes', return_value=([], 30.0)):
                empty = node.run(InputImpl.VideoFromFile(str(source)), 'content', 10.0, 0.0, 1, True)
            self.assertEqual(empty[2], 0)
            self.assertIsInstance(empty[3][0], video_node.ExecutionBlocker)

    def test_scene_text_blocks_zero_scene_prompt_consumers_from_both_detectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / 'source.mp4'
            write_timing_video(source)
            cases = (
                (video_node, video_node.PySceneDetectVideo,
                 (InputImpl.VideoFromFile(str(source)),), 'detect_scenes'),
                (legacy_node, legacy_node.PySceneDetectToImages,
                 (torch.zeros((2, 4, 4, 3)), {'loaded_fps': 30.0}), 'detect_scenes_from_video'),
            )
            for module, cls, args, detector in cases:
                with self.subTest(node=cls.__name__):
                    with patch.object(module, detector, return_value=([], 30.0)):
                        result = cls().run(*args, 'content', 10.0, 0.0, 1, True)
                    # Preserve the existing non-list output contracts and fallback image.
                    self.assertEqual(result[0].shape, (1, 1, 1, 3))
                    self.assertEqual(result[0].sum().item(), 0)
                    self.assertEqual(json.loads(result[1])['scenes'], [])
                    self.assertEqual(result[2], 0)
                    text, prompts = text_node.PySceneDetectSceneText.execute(result[1]).result
                    self.assertEqual(text, '')
                    self.assertEqual(len(prompts), 1)
                    self.assertIsInstance(prompts[0], text_node.ExecutionBlocker)
                    self.assertIsNone(prompts[0].message)

    def test_trimmed_preview_contains_only_active_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output, temp = root / 'output', root / 'temp'
            output.mkdir()
            temp.mkdir()
            source = root / 'source.mp4'
            write_timing_video(source)
            original = source.read_bytes()
            preview = _load_preview_node(str(output), str(temp))
            for start, duration, expected_frames, first, last in (
                (0.5, 1.0, 30, 0, 255),
                (1.0, 0.0, 30, 255, 255),
                (0.0, 0.5, 15, 0, 0),
            ):
                with self.subTest(start=start, duration=duration):
                    video = InputImpl.VideoFromFile(str(source)).as_trimmed(start, duration)
                    entry = preview.preview_entry_for_video(video)
                    saved = temp / entry['subfolder'] / entry['filename']
                    with av.open(str(saved)) as container:
                        frames = list(container.decode(video=0))
                        self.assertEqual(len(frames), expected_frames)
                        self.assertAlmostEqual(float(container.duration / av.time_base), expected_frames / 30, delta=1/30)
                    self.assertAlmostEqual(frames[0].to_ndarray(format='rgb24').mean(), first, delta=3)
                    self.assertAlmostEqual(frames[-1].to_ndarray(format='rgb24').mean(), last, delta=3)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(output.iterdir()), [])

    def test_subtitle_input_clips_keep_video_and_audio_without_subtitles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / 'source.mp4'
            write_timing_video(source)
            subs = root / 'source.srt'
            subs.write_text('1\n00:00:00,000 --> 00:00:02,000\nSubtitle\n')
            mkv = root / 'subtitles.mkv'
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                '-i', str(source), '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=2',
                '-i', str(subs), '-map', '0:v:0', '-map', '1:a:0', '-map', '2:s:0',
                '-c:v', 'libx264', '-bf', '0', '-preset', 'ultrafast',
                '-c:a', 'pcm_s16le', '-c:s', 'ass',
                '-avoid_negative_ts', 'disabled', str(mkv),
            ], check=True, capture_output=True)
            with av.open(str(mkv)) as container:
                self.assertEqual([s.type for s in container.streams], ['video', 'audio', 'subtitle'])
            _, _, count, clips = video_node.PySceneDetectVideo().run(
                InputImpl.VideoFromFile(str(mkv)), 'content', 10.0, 0.0, 1, True,
            )
            self.assertEqual(count, 2)
            for index, clip in enumerate(clips):
                saved = root / f'scene-{index}.mp4'
                clip.save_to(str(saved))
                with av.open(str(saved)) as container:
                    self.assertEqual([s.type for s in container.streams], ['video', 'audio'])
                    self.assertEqual(len(list(container.decode(video=0))), 30)
                with av.open(str(saved)) as container:
                    audio = list(container.decode(audio=0))
                    self.assertGreater(sum(frame.samples for frame in audio), 0)
                    self.assertGreater(max(abs(frame.to_ndarray()).max() for frame in audio), 0)
                    self.assertAlmostEqual(float(audio[0].pts * audio[0].time_base), 0.0, delta=1/48000)
                    self.assertAlmostEqual(float(audio[-1].pts * audio[-1].time_base) + audio[-1].samples / 48000, 1.0, delta=0.025)

    def test_example_rebatches_one_image_per_prompt_in_scene_order(self):
        from comfy_extras.nodes_rebatch import ImageRebatch

        batch = torch.stack([torch.full((4, 4, 3), value) for value in (0.0, 0.5, 1.0)])
        rows = [dict(index=i+1, start_frame=i, end_frame=i+1, duration_frames=1,
                     start_time=str(i), end_time=str(i+1), duration_sec=1.0) for i in range(3)]
        _, prompts = text_node.PySceneDetectSceneText.execute(json.dumps({'scenes': rows}), 'Scene {index}').result
        images = ImageRebatch.execute([batch], [1]).result[0]
        self.assertEqual(len(images), len(prompts))
        self.assertEqual([item.shape[0] for item in images], [1, 1, 1])
        self.assertEqual([(item.mean().item(), text) for item, text in zip(images, prompts)],
                         [(0.0, 'Scene 1'), (0.5, 'Scene 2'), (1.0, 'Scene 3')])
        workflow = json.loads((Path(__file__).resolve().parents[1] / 'workflow/pyscene_workflow_scene_outputs.json').read_text())
        rebatch = next(n for n in workflow['nodes'] if n['type'] == 'RebatchImages')
        self.assertEqual(rebatch['widgets_values'], [1])
        native = next(n for n in workflow['nodes'] if n['type'] == 'PySceneDetectVideo')
        to_rebatch = next(link for link in workflow['links'] if link[0] == rebatch['inputs'][0]['link'])
        self.assertEqual(to_rebatch[1:5], [native['id'], 0, rebatch['id'], 0])
        to_preview = next(link for link in workflow['links'] if link[0] == rebatch['outputs'][0]['links'][0])
        self.assertEqual(to_preview[1:3], [rebatch['id'], 0])
        scene_text = next(n for n in workflow['nodes'] if n['type'] == 'PySceneDetectSceneText')
        to_text = next(link for link in workflow['links'] if link[0] == scene_text['inputs'][0]['link'])
        self.assertEqual(to_text[1:5], [native['id'], 1, scene_text['id'], 0])
        for slot, name in enumerate(('all_scene_prompts', 'per_scene_prompt_list')):
            self.assertEqual(scene_text['outputs'][slot]['name'], name)
            link = next(link for link in workflow['links'] if link[0] == scene_text['outputs'][slot]['links'][0])
            self.assertEqual(link[1:3], [scene_text['id'], slot])
        self.assertEqual(len(native['outputs']), 4)
        self.assertEqual(len(native['widgets_values']), 27)


if __name__ == '__main__':
    unittest.main()
