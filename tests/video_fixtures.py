from fractions import Fraction

import numpy as np


def write_timing_video(path, *, vfr=False, audio=False, short_scenes=False):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 300)
        if audio:
            audio_stream = container.add_stream("aac", rate=48000)
            audio_stream.layout = "stereo"
        values = [0] * 30 + [255] * 30
        if short_scenes:
            values += [128] * 2 + [30] + [200] * 20 + [70]
        for index, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), value, dtype=np.uint8),
                format="rgb24",
            )
            frame.time_base = Fraction(1, 300)
            if vfr:
                frame.pts = index * 30 if index < 30 else 900 + (index - 30) * 10
            else:
                frame.pts = index * 10
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio:
            for index in range(94):
                frame = av.AudioFrame.from_ndarray(
                    np.zeros((2, 1024), dtype=np.float32), format="fltp", layout="stereo",
                )
                frame.sample_rate = 48000
                frame.pts = index * 1024
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
