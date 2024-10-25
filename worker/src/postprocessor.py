import base64
import logging
from typing import List
import tempfile
import numpy as np
from PIL import Image
import os

# 使用 cv2 替代 export_to_video 函数
import cv2

logger = logging.getLogger(__name__)

class VideoGenPostprocessor:
    
    @staticmethod
    def frames_to_temp_mp4(frames, fps=30):
        """
        Convert a list of frames to an MP4 video using a temporary file.
        """
        frames_np = [np.array(frame) for frame in frames]
        height, width, _ = frames_np[0].shape

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            video_writer = cv2.VideoWriter(temp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for frame in frames_np:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()

        return temp_video_file.name

    @staticmethod
    def mp4_to_base64(mp4_path):
        """
        Convert MP4 file to Base64-encoded string.
        """
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