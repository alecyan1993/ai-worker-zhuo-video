from pathlib import Path
import sys

import base64
import io
import pytest
from PIL import Image
from diffusers.utils import export_to_gif

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "worker"))

from handler import VideoGenHandler
from src.input import VideoGenInput
from src.model_loader import ModelLoader
from src.postprocessor import VideoGenPostprocessor
from src.preprocessor import VideoGenPreprocessor


def base64_to_mp4(base64_string, output_path="output_video.mp4"):
    """
    Convert a Base64-encoded string back to an MP4 file.
    
    Args:
        base64_string (str): The Base64-encoded MP4 string.
        output_path (str): The file path to save the MP4 file.
    """
    # 解码 Base64 字符串为二进制数据
    video_data = base64.b64decode(base64_string)
    
    # 将二进制数据写入文件
    with open(output_path, "wb") as video_file:
        video_file.write(video_data)


def test_video_gen():
    handler = VideoGenHandler()

    # initialize of handler
    model_loader = ModelLoader()
    handler._pipeline = model_loader.load_diff_pipeline()

    image = Image.open(Path(__file__).parent.parent / f"images/demo2.png")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    raw_input_data = {"image_base64": image_b64}

    input_data = VideoGenInput(**raw_input_data)

    image_base64 = input_data.image_base64

    handler.image = VideoGenPreprocessor.get_image_from_base64(image_base64)

    output = handler._pipeline(
        prompt=input_data.prompt,
        negative_prompt=input_data.negative_prompt,
        num_frames=input_data.num_frames,
        width=input_data.width,
        height=input_data.height,
        ip_adapter_image=handler.image,
        guidance_scale=input_data.guidance_scale,
        num_inference_steps=input_data.num_inference_steps,
    )
    frames = output.frames[0]
    export_to_gif(frames, "ex.gif")


    base64_video = VideoGenPostprocessor.convert_frames_to_base64(frames)
    # base64_to_mp4(base64_video, "output_video.mp4")

