import re
import pandas as pd
from llama_cpp import Llama
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
# Load the generator model
generator_model_path = r'C:/dev/python-model/eeve/ggml-model-Q5_K_M.gguf'
llama = Llama(model_path=generator_model_path, n_ctx=4096)
def generate_text_stream(prompt, max_new_tokens=700):
    for token in llama(prompt, max_tokens=max_new_tokens, temperature=0.08, stream=True):
        yield token['choices'][0]['text']

def create_prompt(search_results, new_case_info):
    prompt = '''
          당신은 대한민국의 법률 전문가입니다. 아래 주어진 현재 사건과 유사도가 높은 판례들을 참고하여 현재 사건에 대한 예상 형량을 예측해 주세요.
          주어진 판례들만 고려하고, 외부 정보는 사용하지 마세요.
          처음과 끝에 **로 둘러싸인 부분을 큰 글씨로 강조하여 표시해 주세요.

          예상 형량은 다음 중 하나의 형식으로 정확히 명시해 주세요:
          - 무죄일 경우 : 무죄
          - 집행유예일 경우 : 징역 XX년/개월 집행유예 XX년/개월
          - 벌금일 경우 : 벌금 XX원
          - 징역일 경우 : 징역 XX년/개월

          응답에 다음 항목을 반드시 포함하고 명확하게 이야기하세요.:
          **1. 예측 형량:** (유사 주문들을 참고하여 예상 형량을 명확하게 제시)
          2. 분석: (예상 형량 도출 이유 설명 , 유사 판례들의 판결 내용과 양형 이유 참고)
          3. 법률적 조언: (피고인의 현재 상황에 대한 간단한 법률적 조언 제공)
          '''

    if isinstance(search_results, list):
        search_results = pd.DataFrame(search_results)

    for i, row in search_results.iterrows():
        prompt += f"유사 사건 {i + 1} [유사 점수(bm25스코어*0.5 + 코사인유사도*0.5) : {row['combinedScore']:.2f}]\n{row['주문']}\n{row['양형의 이유']}\n{row['전문']}\n\n"

    prompt += f"현재 사건: {new_case_info}\n"
    return prompt

def generate_sentence_stream(search_results, new_case_info, max_new_tokens=700):
    prompt = create_prompt(search_results, new_case_info)
    print(f"Prompt: {prompt}")
    for token in generate_text_stream(prompt, max_new_tokens):
        yield token
        print(token, end='', flush=True)  # 디버깅을 위해 토큰 출력
    print()  # 줄바꿈
