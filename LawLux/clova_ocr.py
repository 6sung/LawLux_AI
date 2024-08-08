import time
import pandas as pd
import cv2
import json
import matplotlib.pyplot as plt

import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image

import requests
import uuid
import time
import json

api_url = ' '
secret_key = ' '

def ocr(file):
    file.seek(0)  # 파일 포인터를 처음으로 되돌리기
    file_content = file.read()
    print(f"File size: {len(file_content)} bytes")  # 파일 크기 확인
    file.seek(0)  # 파일 포인터를 다시 처음으로 되돌리기

    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    files = {
        'file': (file.filename, file, file.content_type)
    }

    headers = {'X-OCR-SECRET': secret_key}
    response = requests.post(api_url, headers=headers, data=payload, files=files)

    # Print response text for debugging
    print(response.text)

    try:
        data = json.loads(response.text)
        # Adjust based on the actual response format
        text_lines = [field['inferText'] for image in data['images'] for field in image['fields']]
        output_text = ' '.join(text_lines)
        return output_text
    except KeyError as e:
        raise Exception(f"Missing key in response data: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON response: {e}")
