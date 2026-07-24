"""Runpod serverless handler for FLUX.1-dev image generation."""

import os
import base64
import io
import time
from typing import Any, Dict

import torch

import runpod


# Set HF_TOKEN in your Runpod environment for gated model access.
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Optional performance/quality defaults
DEFAULT_NUM_INFERENCE_STEPS = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", "28"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "3.5"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "-1"))

# Singleton pipeline (loaded lazily or preloaded in __main__)
_pipeline = None


def _load_pipeline():
    """Lazy-load the FLUX.1-dev Diffusers pipeline."""
    from diffusers import FluxPipeline

    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("Loading FLUX.1-dev pipeline...")
    start = time.time()
    _pipeline = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    # Offload to CPU when idle to reduce VRAM on Runpod serverless
    _pipeline.enable_model_cpu_offload()
    # VAE slicing helps with memory for larger resolutions
    _pipeline.enable_vae_slicing()
    print(f"Pipeline loaded in {time.time() - start:.2f}s")
    return _pipeline


def _get_input(job_input: Dict[str, Any], key: str, default=None):
    """Fetch a value from the job input, supporting nested 'input' key."""
    if "input" in job_input and isinstance(job_input["input"], dict):
        return job_input["input"].get(key, default)
    return job_input.get(key, default)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Runpod serverless handler entry point."""
    job_input = job.get("input", job)

    prompt = _get_input(job_input, "prompt")
    if not prompt or not isinstance(prompt, str):
        return {
            "error": "Missing or invalid 'prompt'. Provide a non-empty text prompt."
        }

    negative_prompt = _get_input(job_input, "negative_prompt", "")
    num_inference_steps = int(_get_input(job_input, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    guidance_scale = float(_get_input(job_input, "guidance_scale", DEFAULT_GUIDANCE_SCALE))
    height = int(_get_input(job_input, "height", DEFAULT_HEIGHT))
    width = int(_get_input(job_input, "width", DEFAULT_WIDTH))
    seed = int(_get_input(job_input, "seed", DEFAULT_SEED))

    if seed < 0:
        seed = int(time.time() * 1000) % (2**32)

    pipe = _load_pipeline()
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    start = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    inference_time = time.time() - start

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "image": img_base64,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "seed": seed,
        "inference_time_seconds": round(inference_time, 2),
    }


if __name__ == "__main__":
    # Preload model before starting the serverless loop.
    _load_pipeline()
    runpod.serverless.start({"handler": handler})
