import base64
import tempfile
import os
from typing import List
from PIL import Image
from diffusers.utils import export_to_video

class VideoGenPostprocessor:
    
    @staticmethod
    def frames_to_temp_mp4(frames, fps=8):
        """
        Convert a list of frames to an MP4 video and return the temporary file path.

        Args:
            frames (list): List of frames (each frame is a PIL.Image object).
            fps (int): Frames per second for the video.

        Returns:
            str: Path to the temporary MP4 file.
        """
        # 创建一个临时文件保存 MP4 视频
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            # 使用 export_to_video 函数将 frames 转换为 MP4 文件
            export_to_video(frames, output_video_path=temp_video_file.name, fps=fps)
        
        return temp_video_file.name

    @staticmethod
    def mp4_to_base64(mp4_path):
        """
        Convert MP4 file to Base64-encoded string.

        Args:
            mp4_path (str): Path to the MP4 file.

        Returns:
            str: Base64 encoded MP4 video string.
        """
        with open(mp4_path, "rb") as video_file:
            video_data = video_file.read()
            base64_encoded = base64.b64encode(video_data).decode("utf-8")
        return base64_encoded

    @staticmethod
    def convert_frames_to_base64(frames: List[Image.Image], fps: int = 8):
        """
        Convert a list of frames to a Base64-encoded MP4 video.

        Args:
            frames (List[Image.Image]): List of frames to encode.
            fps (int): Frames per second for the video.

        Returns:
            str: Base64 encoded MP4 video string.
        """
        # 将 frames 转换为临时 MP4 文件路径
        temp_video_path = VideoGenPostprocessor.frames_to_temp_mp4(frames, fps)
        
        # 将临时 MP4 文件编码为 Base64
        base64_encoded = VideoGenPostprocessor.mp4_to_base64(temp_video_path)
        
        # 删除临时 MP4 文件
        os.remove(temp_video_path)
        
        return base64_encoded