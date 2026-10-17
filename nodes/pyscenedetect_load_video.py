from __future__ import annotations

import os

import folder_paths
import torch
from scenedetect import open_video

from ..utils.video_ops import frame_to_tensor_bhwc


class PySceneDetectLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "backend": (["pyav", "opencv"], {"default": "pyav"}),
            },
            "optional": {
                "load_audio": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
    FUNCTION = "load_video"
    CATEGORY = "Video/PySceneDetect"

    def load_video(self, video: str, backend: str = "pyav", load_audio: bool = True):
        path = folder_paths.get_annotated_filepath(video)
        stream = open_video(path, backend=backend)
        fps = float(stream.frame_rate)
        width, height = stream.frame_size
        duration = stream.duration
        frames, timecodes = [], []
        while True:
            frame = stream.read()
            if frame is False:
                break
            frames.append(frame_to_tensor_bhwc(frame))
            timecodes.append(stream.position)
        if not frames:
            raise ValueError("No video frames decoded.")

        images = torch.cat(frames, dim=0)
        video_info = {
            "source_fps": fps,
            "source_frame_count": duration.frame_num,
            "source_duration": duration.seconds,
            "source_width": width,
            "source_height": height,
            "loaded_fps": fps,
            "loaded_frame_count": len(images),
            "loaded_duration": duration.seconds,
            "loaded_width": int(images.shape[2]),
            "loaded_height": int(images.shape[1]),
            # Runtime-only timing accompanies the batch; never reconstruct PTS from FPS.
            "scenedetect_timecodes": tuple(timecodes),
            "scenedetect_end_position": stream.position,
        }
        del stream, frames

        audio = None
        if load_audio:
            # Audio is optional; the video backend and its fallback remain upstream-owned.
            import av
            with av.open(path) as container:
                has_audio = bool(container.streams.audio)
            if has_audio:
                from comfy_extras.nodes_audio import load

                waveform, sample_rate = load(path)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

        return images, len(images), audio, video_info

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return os.path.getmtime(folder_paths.get_annotated_filepath(video))

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


NODE_CLASS_MAPPINGS = {"PySceneDetectLoadVideo": PySceneDetectLoadVideo}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PySceneDetectLoadVideo": "PySceneDetect: Load Video (VHS Compatible)"
}
