import base64
import io
import logging
from typing import Tuple
from pathlib import Path
import yaml
import time
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class VideoGenPreprocessor:

    def get_image_from_base64(self, image_base64: str, image_format: str = "png"):
        """
        Function to generate image from base64
        """
        image_data = base64.b64decode(image_base64)
        image_io = io.BytesIO(image_data)
        image = Image.open(image_io)

        # convert to rgb
        image = image.convert("RGB")
        return image_io
        