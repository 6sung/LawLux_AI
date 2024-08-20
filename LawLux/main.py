import os

from clova_ocr import ocr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import base64
from flask import Flask, request, render_template, jsonify, Response
from classify import AIModule
from rag_search import search_query
from rag_generate import generate_sentence

app = Flask(__name__)

model_path = r'C:/dev/python-model/bert-kor-base-lawQA'
ai_module = AIModule(model_path)

@app.route('/')
def index():
    return render_template('index.html', message='')


from flask import stream_with_context


@app.route('/ai_service', methods=['POST'])
def ai_service():
    try:
        input_message = request.form.get('message', '')
        print(f"Received message: {input_message}")

        file = request.files.get('file')
        if file:
            print(f"File size: {len(file.read())} bytes")
            print(f"Received file: {file.filename}")
            file_content = file.read()
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            file_info = {
                'filename': file.filename,
                'filetype': file.content_type
            }
            message = input_message
        else:
            message = input_message

        print(message)
        query_classify = ai_module.predict(message)
        print(f"질문 분류 결과: {query_classify}")

        def generate():
            if query_classify == '법률질문':
                search_results = search_query(message)
                print(f"Search results: {search_results}")
                new_case_info = message

                for chunk in generate_sentence(search_results, new_case_info):
                    print(f'main_chunk', chunk)
                    yield chunk
            else:
                yield query_classify

        return Response(stream_with_context(generate()), mimetype='application/json; charset=utf-8')

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Content-Type', 'application/json; charset=utf-8')
    return response

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
