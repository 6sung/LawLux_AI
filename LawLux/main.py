import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, render_template, jsonify, Response
from classify import AIModule
from rag_search import search_query
import json
#from rag_generate import generate_sentence

app = Flask(__name__)

model_path = r'C:/dev/python-model/bert-kor-base-lawQA'
ai_module = AIModule(model_path)

@app.route('/')
def index():
    return render_template('index.html', message='')

@app.route('/ai_service', methods=['POST'])
def ai_service():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '')
        print(f"Received message: {message}")

        query_classify = ai_module.predict(message)
        print(f"질문 분류 결과: {query_classify}")

        if query_classify == '법률질문':
            search_results = search_query(message)
            print(search_results)
            response_message = search_results
            #new_case_info = message
            #gen_sentence = generate_sentence(search_results, new_case_info)
            #response_message = f"예상 형량: {gen_sentence}"
        else:
            response_message = f"{query_classify}"

        #return jsonify({'message' : search_results})
        #return jsonify({'message': response_message})
        response_json = json.dumps(response_message, ensure_ascii=False)  # ensure_ascii=False로 설정하여 한글이 깨지지 않도록 함

        return Response(response=response_json, status=200, mimetype='application/json; charset=utf-8')

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
    app.run(host='localhost', port=5000, debug=False)
