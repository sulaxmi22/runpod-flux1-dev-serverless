"""Simple client to test a deployed Runpod serverless endpoint."""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Test a Runpod serverless endpoint for FLUX.1-dev")
    parser.add_argument("--endpoint-id", required=True, help="Runpod endpoint ID")
    parser.add_argument("--api-key", required=True, help="Runpod API key")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--output", default="generated_image.png", help="Output image filename")
    parser.add_argument("--poll-interval", type=int, default=2, help="Seconds between status polls")
    args = parser.parse_args()

    run_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/run"
    status_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/status"

    payload = {
        "input": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "seed": args.seed,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    print("Submitting job...")
    response = requests.post(run_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    job_id = data["id"]
    print(f"Job submitted: {job_id}")

    while True:
        status_response = requests.get(f"{status_url}/{job_id}", headers=headers, timeout=30)
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data.get("status")
        print(f"Status: {status}")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            if "error" in output:
                print(f"Endpoint returned error: {output['error']}", file=sys.stderr)
                sys.exit(1)

            image_b64 = output.get("image")
            if not image_b64:
                print("No image in response:", json.dumps(status_data, indent=2))
                sys.exit(1)

            image_bytes = base64.b64decode(image_b64)
            output_path = Path(args.output)
            output_path.write_bytes(image_bytes)
            print(f"Image saved to {output_path.resolve()}")
            print(f"Inference time: {output.get('inference_time_seconds', 'N/A')}s")
            break

        if status in ("FAILED", "TIMED_OUT", "CANCELLED"):
            print(f"Job ended with status {status}:", json.dumps(status_data, indent=2), file=sys.stderr)
            sys.exit(1)

        import time
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
