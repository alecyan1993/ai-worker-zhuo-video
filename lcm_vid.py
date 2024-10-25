import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, load_image, export_to_video

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained(
    "emilianJR/epiCRealism", motion_adapter=adapter
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights(
    "wangfuyun/AnimateLCM",
    weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
    adapter_name="lcm-lora",
)
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out"
)
pipe.set_adapters(["lcm-lora", "zoom-out"], [1.0, 0.8])


pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.safetensors"
)

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

ipa_img = load_image("demo3.png")

output = pipe(
    prompt="moving image, video, animation, film, movie, motion picture",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    width=640,
    height=416,
    ip_adapter_image=ipa_img,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animatelcm.gif")
export_to_video(frames, "animatelcm.mp4")
