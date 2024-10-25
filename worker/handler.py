import logging
import os
import time
from pathlib import Path
from typing import List, Optional
from diffusers import AnimateDiffPipeline

from PIL import Image
from src.input import VideoGenInput
from src.model_loader import ModelLoader
from src.preprocessor import VideoGenPreprocessor
from src.postprocessor import VideoGenPostprocessor
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler
from utils.exception import InferenceError, PostprocessingError, PreprocessingError


logger = logging.getLogger(__name__)


class VideoGenHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._pipeline: Optional[AnimateDiffPipeline] = None
        self._video_gen_preprocessor: Optional[VideoGenPreprocessor] = None
        self._video_gen_postprocessor: Optional[VideoGenPostprocessor] = None
        self.image: Optional[Image.Image] = None

    def initialize(self, context):
        logger.debug(
            f"initalize: context type: {type(context)}, context = '{dir(context)}'"
        )

        manifest = context.manifest
        logger.info(f"manifest: {manifest}")

        properties = context.system_properties
        logger.info(f"properties: {properties}")

        gpu_id = properties.get("gpu_id")
        logger.info(f"gpu_id: {gpu_id}")
        logger.info(f"cwd '{os.getcwd()}'")

        model_loader = ModelLoader()

        self._pipeline = model_loader.load_diff_pipeline()

        self._video_gen_preprocessor = VideoGenPreprocessor()
        self._video_gen_postprocessor = VideoGenPostprocessor()
        logger.info("Torchserve backend worker initialized")

    def preprocess(self, data: List[dict]):
        raw_input_data: dict = data[0]["body"]

        logger.info("Validating raw input data")
        input_data = VideoGenInput(**raw_input_data)

        logger.info(f"Input data: {raw_input_data.keys()}")

        image_base64 = input_data.image_base64
        self.image = VideoGenPreprocessor.get_image_from_base64(image_base64)

        return input_data

    def inference(self, input_data: VideoGenInput):
        output = self._pipeline(
            prompt=input_data.prompt,
            negative_prompt=input_data.negative_prompt,
            num_frames=input_data.num_frames,
            width=input_data.width,
            height=input_data.height,
            ip_adapter_image=self.image,
            guidance_scale=input_data.guidance_scale,
            num_inference_steps=input_data.num_inference_steps,
        )
        frames = output.frames[0]
        return frames

    def postprocess(self, inference_output):
        base64_video = VideoGenPostprocessor.convert_frames_to_base64(inference_output)
        response = [{"video": base64_video}]
        return response

    def handle(self, data, context):
        overall_start = time.time()

        try:
            pre_start = time.time()
            model_input = self.preprocess(data)
            logger.info(f"Preprocessing duration: {time.time() - pre_start:.2f} sec")
        except Exception as error:
            raise PreprocessingError("Preprocessing failed") from error

        try:
            inf_start = time.time()
            model_output = self.inference(model_input)
            logger.info(f"Inference duration: {time.time() - inf_start:.2f} sec")
        except Exception as error:
            raise InferenceError("Inference failed") from error

        try:
            post_start = time.time()
            response = self.postprocess(model_output)
            logger.info(f"Post-processing duration: {time.time() - post_start:.2f} sec")
        except Exception as error:
            raise PostprocessingError("Postprocessing failed") from error

        logger.info(f"Overall Process duration: {time.time() - overall_start:.2f} sec")

        return response
