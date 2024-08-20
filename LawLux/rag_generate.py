import re
import time

import pandas as pd
from llama_cpp import Llama
import sys

# Load the generator model
generator_model_path = r'C:/dev/python-model/eeve/ggml-model-Q5_K_M.gguf'
llama = Llama(model_path=generator_model_path, n_ctx=4096)

# 1. 프롬프트 생성
def create_prompt(search_results, new_case_info):
    prompt = '''
           당신은 대한민국의 법률 전문가입니다. 아래 주어진 현재 사건과 유사도가 높은 판례들을 참고하여 현재 사건에 대한 예상 형량을 예측해 주세요. 
           주어진 판례들만 고려하고, 외부 정보는 사용하지 마세요.

           예상 형량은 다음 중 하나로 정확히 명시해 주세요:
           - 무죄
           - 징역 XX년/개월 집행유예 XX년/개월
           - 벌금 XX원
           - 징역 XX년/개월

           응답에 다음 항목을 반드시 포함하세요:
           1. 예측 형량: 유사 주문들을 참고하여 예상 형량을 명확하게 제시
           2. 분석: 예상 형량 도출 이유 설명 (유사 판례들의 판결 내용(전문)과 양형 이유 참고)
           3. 양형의 이유: 유사 판례들의 양형 이유를 종합하여 현재 사건에 적용 가능한 양형 이유 정리
           4. 법률적 조언: 피고인의 현재 상황에 대한 간단한 법률적 조언 제공

           아래는 유사한 판례들입니다:
       '''

    if isinstance(search_results, list):
        search_results = pd.DataFrame(search_results)

    for i, row in search_results.iterrows():
        prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{row['양형의 이유']}\n\n"

    prompt += f"현재 사건: {new_case_info}\n\n"
    prompt += "예상 형량:"

    return prompt

# 2. Llama 모델 로드 및 텍스트 생성
def generate_text(prompt, max_new_tokens=256):
    stream = llama(prompt, max_tokens=max_new_tokens, temperature=0.8, stream=True)
    for chunk in stream:
        yield chunk['choices'][0]['text']

# 3. 생성된 텍스트에서 정보 추출 및 형식화
def generate_sentence(search_results, new_case_info, max_new_tokens=512):
    prompt = create_prompt(search_results, new_case_info)
    for chunk in generate_text(prompt, max_new_tokens):
        #print(f'generate_chunk', chunk)
        yield chunk
