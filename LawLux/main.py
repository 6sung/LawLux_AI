import os
from clova_ocr import ocr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import base64
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from classify import AIModule
from rag_search import hybrid_cc
from rag_generate import generate_sentence_stream

app = Flask(__name__)
model_path = r'C:/dev/python-model/bert-kor-base-lawQA'
ai_module = AIModule(model_path)


@app.route('/')
def index():
    return render_template('index.html', message='')


@app.route('/ai_service', methods=['POST'])
def ai_service():
    def generate():
        try:
            input_message = request.form.get('message', '')
            print(f"Received message: {input_message}")
            file = request.files.get('file')
            if file:
                print(f"Received file: {file.filename}")
                file_content = file.read()
                print(f"File size: {len(file_content)} bytes")
                file.seek(0)  # 파일 포인터를 처음으로 되돌림
                ocr_message = ocr(file)
                print(ocr_message)
                message = ocr_message + input_message
            else:
                message = input_message
            print(message)
            query_classify = ai_module.predict(message)
            print(f"질문 분류 결과: {query_classify}")

            if query_classify == '법률질문':
                search_results = hybrid_cc(message)
                print(f"Search results: {search_results}")
                new_case_info = message

                def truncate_text(text, max_length=160):
                    if isinstance(text, str):
                        return text[:max_length] + ('...' if len(text) > max_length else '')
                    return text

                for result in search_results:
                    result["전문"] = truncate_text(result["전문"])

                # 검색 결과 먼저 전송
                yield json.dumps({'search_results': search_results}, ensure_ascii=False) + '\n'

                # 생성 문장을 토큰 단위로 스트리밍
                for token in generate_sentence_stream(search_results, new_case_info):
                    yield json.dumps({'token': token}, ensure_ascii=False) + '\n'
            else:
                yield json.dumps({'message': f"{query_classify}"}, ensure_ascii=False) + '\n'

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            yield json.dumps({'error': str(e)}, ensure_ascii=False) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/json')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Content-Type', 'application/json; charset=utf-8')
    return response


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
