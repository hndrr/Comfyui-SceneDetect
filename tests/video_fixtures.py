from fractions import Fraction

import numpy as np


def write_timing_video(path, *, vfr=False):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 300)
        for index in range(60):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), 0 if index < 30 else 255, dtype=np.uint8),
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
