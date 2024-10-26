import base64
import tempfile
import os
from typing import List
from PIL import Image
import ffmpeg
import numpy as np

class VideoGenPostprocessor:

    @staticmethod
    def frames_to_temp_mp4(frames: List[Image.Image], fps=8):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            temp_video_path = temp_video_file.name

        # 将帧列表转换为 numpy 数组格式，并将每一帧从 PIL 转换为 RGB
        frame_arrays = [np.array(frame.convert("RGB")) for frame in frames]

        # 使用 ffmpeg-python 处理视频，直接生成 H.264 编码的视频
        (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{frame_arrays[0].shape[1]}x{frame_arrays[0].shape[0]}', framerate=fps)
            .output(temp_video_path, vcodec='libx264', crf=23, preset='fast')
            .run(input=np.concatenate(frame_arrays).tobytes(), overwrite_output=True)
        )

        return temp_video_path

    @staticmethod
    def mp4_to_base64(mp4_path):
        with open(mp4_path, "rb") as video_file:
            video_data = video_file.read()
            base64_encoded = base64.b64encode(video_data).decode("utf-8")
        return base64_encoded

    @staticmethod
    def convert_frames_to_base64(frames: List[Image.Image], fps: int = 8):
        temp_video_path = VideoGenPostprocessor.frames_to_temp_mp4(frames, fps)
        base64_encoded = VideoGenPostprocessor.mp4_to_base64(temp_video_path)
        os.remove(temp_video_path)
        return base64_encoded