import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import imageio_ffmpeg
import torch
from scenedetect import FrameTimecode

from test_preview_node import _load_preview_node
from test_video_node import InputImpl, video_node
from video_fixtures import write_timing_video
from utils.scene_clips import split_scene_clips


class ReviewRegressionTests(unittest.TestCase):
    def test_disabled_and_zero_scene_outputs_skip_video_consumers_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / 'source.mp4'
            write_timing_video(source)
            node = video_node.PySceneDetectVideo()
            result = node.run(InputImpl.VideoFromFile(str(source)), 'content', 10.0, 0.0, 1, True)
            self.assertEqual(result[0].shape[0], 2)
            self.assertEqual(result[2], 2)
            self.assertEqual(len(result[5]), 2)
            self.assertEqual(len(result[3]), 1)
            self.assertIsInstance(result[3][0], video_node.ExecutionBlocker)
            self.assertIsNone(result[3][0].message)
            with patch.object(video_node, 'detect_scenes', return_value=([], 30.0)):
                empty = node.run(InputImpl.VideoFromFile(str(source)), 'content', 10.0, 0.0, 1, True, split_clips=True)
            self.assertEqual(empty[2], 0)
            self.assertIsInstance(empty[3][0], video_node.ExecutionBlocker)

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
            scenes = [(FrameTimecode(0, 30.0), FrameTimecode(60, 30.0))]
            for reencode in (False, True):
                with self.subTest(reencode=reencode):
                    clips = split_scene_clips(str(mkv), scenes, str(root / str(reencode)), reencode=reencode)
                    self.assertEqual(len(clips), 1)
                    with av.open(clips[0]) as container:
                        self.assertEqual([s.type for s in container.streams], ['video', 'audio'])
                        self.assertEqual(len(list(container.decode(video=0))), 60)
                    with av.open(clips[0]) as container:
                        audio = list(container.decode(audio=0))
                        self.assertGreater(sum(frame.samples for frame in audio), 0)
                        self.assertGreater(max(abs(frame.to_ndarray()).max() for frame in audio), 0)

    def test_example_rebatches_one_image_per_prompt_in_scene_order(self):
        from comfy_extras.nodes_rebatch import ImageRebatch
        from utils.scene_text import format_scenes_for_llm

        batch = torch.stack([torch.full((4, 4, 3), value) for value in (0.0, 0.5, 1.0)])
        rows = [dict(index=i+1, start_frame=i, end_frame=i+1, duration_frames=1,
                     start_time=str(i), end_time=str(i+1), duration_sec=1.0) for i in range(3)]
        _, prompts = format_scenes_for_llm(rows, 'Scene {index}')
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


if __name__ == '__main__':
    unittest.main()
