"""Runpod Flash (no-Docker) endpoint for FLUX.1-dev image generation."""

import os
import base64
import io
import time

from runpod_flash import Endpoint, GpuGroup


# Pipeline singleton kept alive inside a worker across requests.
_pipeline = None


def _get_pipeline():
    """Lazy-load the FLUX.1-dev Diffusers pipeline once per worker."""
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import FluxPipeline

        model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
        token = os.getenv("HF_TOKEN", None)

        print("Loading FLUX.1-dev pipeline...")
        start = time.time()
        _pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        # 24 GB GPU friendly: keep weights on CPU until needed, then slice VAE.
        _pipeline.enable_model_cpu_offload()
        _pipeline.enable_vae_slicing()
        print(f"Pipeline loaded in {time.time() - start:.2f}s")

    return _pipeline


@Endpoint(
    name="flux1-dev-flash",
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_24, GpuGroup.AMPERE_80],
    workers=(0, 2),
    idle_timeout=900,
    dependencies=[
        "diffusers>=0.30.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece",
        "Pillow",
        "safetensors",
        "huggingface-hub",
        "numpy<2.0",
    ],
    env={
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "MODEL_ID": os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev"),
    },
)
async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = -1,
) -> dict:
    """Generate an image from a text prompt using FLUX.1-dev."""
    if not prompt:
        return {"error": "Missing or empty 'prompt' parameter."}

    import torch

    if seed < 0:
        seed = int(time.time() * 1000) % (2**32)

    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    pipe = _get_pipeline()
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
    # Local manual test (sends the job to the Flash endpoint when deployed).
    import asyncio
    result = asyncio.run(generate_image(prompt="A futuristic city at sunset, cyberpunk, highly detailed"))
    print(result)
