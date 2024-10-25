from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    StrictInt,
    StrictStr,
)


class VideoGenInput(BaseModel):
    image_base64: StrictStr = Field(description="image as base64 string", default=None)
    prompt: Optional[StrictStr] = Field(
        description="prompt for the generation",
        default="moving image, video, animation, film, movie, motion picture",
    )
    negative_prompt: Optional[StrictStr] = Field(
        description="negative prompt for the generation",
        default="unfocused, blurry, grainy, low quality, bad quality, worse quality, low resolution",
    )
    width: Optional[StrictInt] = Field(
        description="width of the generated image", default=640
    )
    height: Optional[StrictInt] = Field(
        description="height of the generated image", default=416
    )
    num_frames: Optional[StrictInt] = Field(
        description="number of frames in the generated video", default=16
    )
    guidance_scale: Optional[float] = Field(
        description="guidance scale for the generation", default=1.5
    )
    num_inference_steps: Optional[StrictInt] = Field(
        description="number of inference steps", default=6
    )
