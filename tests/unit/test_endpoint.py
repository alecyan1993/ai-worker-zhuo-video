from pathlib import Path
import sys
import requests
from PIL import Image
from io import BytesIO
import cv2
from PIL import Image
import imageio
import base64

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "worker"))


def mp4_to_gif(mp4_path, gif_path="output.gif", fps=10):
    """
    Convert an MP4 file to a GIF.

    Args:
        mp4_path (str): Path to the MP4 file.
        gif_path (str): Path to save the output GIF.
        fps (int): Frames per second for the GIF.
    """
    # 使用 OpenCV 读取 MP4 文件
    cap = cv2.VideoCapture(mp4_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换 BGR（OpenCV 默认）为 RGB（PIL 需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    # 将帧保存为 GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # 持续时间是以毫秒为单位
        loop=0,
    )


def base64_to_mp4(base64_string, output_path="output_video.mp4"):
    """
    Convert a Base64-encoded string back to an MP4 file.

    Args:
        base64_string (str): The Base64-encoded MP4 string.
        output_path (str): The file path to save the MP4 file.
    """
    video_data = base64.b64decode(base64_string)

    with open(output_path, "wb") as video_file:
        video_file.write(video_data)


def test_endpoint():
    url = "http://150.136.215.49:8083/predictions/video_gen_endpoint"
    # read voice file
    image = Image.open(Path(__file__).parent.parent / f"images/demo2.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # preprocess of handler
    data = {"image_base64": image_b64}

    response = requests.post(url, json=data)

    res = response.json()
    print(f"Results Keys: {res.keys()}")

    base64_video = res["video"]
    print(f"Length of Base64 video data: {len(base64_video)}")
    base64_to_mp4(base64_video, "output.mp4")
    mp4_to_gif("output.mp4", "output.gif", fps=8)
