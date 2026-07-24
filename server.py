import os
import json
import time
import requests
from flask import Flask, request, Response, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID')
POLL_INTERVAL = float(os.environ.get('POLL_INTERVAL', '2'))


@app.route('/')
def index():
    return send_from_directory('ui', 'index.html')


@app.route('/generate', methods=['POST'])
def generate():
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        return {'error': 'Server not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID.'}, 500

    data = request.get_json(force=True) or {}
    payload = {
        'input': {
            'prompt': data.get('prompt', ''),
            'negative_prompt': data.get('negative_prompt', ''),
            'width': int(data.get('width', 1024)),
            'height': int(data.get('height', 1024)),
            'num_inference_steps': int(data.get('num_inference_steps', 28)),
            'guidance_scale': float(data.get('guidance_scale', 3.5)),
            'seed': int(data.get('seed', -1)),
        }
    }

    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + RUNPOD_API_KEY}
    submit_url = 'https://api.runpod.ai/v2/' + RUNPOD_ENDPOINT_ID + '/run'

    try:
        submit_resp = requests.post(submit_url, headers=headers, json=payload, timeout=30)
        submit_resp.raise_for_status()
        job_id = submit_resp.json()['id']
    except Exception as exc:
        return {'error': 'Submit failed: ' + str(exc)}, 502

    return {'job_id': job_id}


@app.route('/status/<job_id>')
def status_stream(job_id):
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        def missing():
            nl = chr(10)
            yield 'event: status' + nl + 'data: ' + json.dumps({'status': 'ERROR: Server not configured', 'error': 'Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in the .env file and restart the server.'}) + nl + nl
        return Response(missing(), mimetype='text/event-stream')

    def stream():
        nl = chr(10)
        status_url = 'https://api.runpod.ai/v2/' + RUNPOD_ENDPOINT_ID + '/status/' + job_id
        auth_header = {'Authorization': 'Bearer ' + RUNPOD_API_KEY}

        while True:
            try:
                resp = requests.get(status_url, headers=auth_header, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                status = data.get('status')

                if status == 'COMPLETED':
                    out = data.get('output', {})
                    if out.get('error'):
                        yield 'event: status' + nl + 'data: ' + json.dumps({'status': 'FAILED: ' + out['error'], 'error': out['error']}) + nl + nl
                    else:
                        yield 'event: status' + nl + 'data: ' + json.dumps({'status': 'COMPLETED', 'image': out.get('image'), 'inference_time_seconds': out.get('inference_time_seconds')}) + nl + nl
                    return

                if status in ('FAILED', 'TIMED_OUT', 'CANCELLED'):
                    yield 'event: status' + nl + 'data: ' + json.dumps({'status': 'Job ' + status, 'error': status}) + nl + nl
                    return

                yield 'event: status' + nl + 'data: ' + json.dumps({'status': status}) + nl + nl

            except Exception as exc:
                yield 'event: status' + nl + 'data: ' + json.dumps({'status': 'ERROR: ' + str(exc), 'error': str(exc)}) + nl + nl
                return

            time.sleep(POLL_INTERVAL)

    return Response(stream(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5001')), debug=False)
