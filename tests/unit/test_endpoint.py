from pathlib import Path
import sys
import requests
from PIL import Image
from io import BytesIO

import base64

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "worker"))


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
    base64_to_mp4(base64_video, "output.mp4")
