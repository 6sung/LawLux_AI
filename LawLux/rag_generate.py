import re
import pandas as pd
from llama_cpp import Llama
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# Load the generator model
#generator_model_path = r'D:/dev/python-model/qwen/qwen2-7b-instruct-q5_k_m.gguf' # qwen
generator_model_path = r'C:/dev/python-model/eeve/ggml-model-Q5_K_M.gguf' # eeve
#generator_model_path = r'D:/dev/python-model/llama3_luxia/Ko-Llama3-Luxia-8B.Q5_K_M.gguf' # llama3_luxia
llama = Llama(model_path=generator_model_path, n_ctx=4096)

def summarize_text(text, max_length=1024):
    prompt = f"다음 법률 텍스트를 {max_length}자 내외로 요약해주세요 주어진 내용만 고려하고, 외부 정보는 사용하지 마세요.:\n\n{text}\n\n요약:"
    response = llama(prompt, max_tokens=max_length, temperature=0.7)
    summary = response['choices'][0]['text'].strip()
    return summary

def summarize_generate_text(prompt, max_new_tokens=512):
    response = llama(prompt,
                     max_tokens=max_new_tokens,
                     temperature=0.7,
                     top_p=0.9,
                     frequency_penalty=0.2,
                     presence_penalty=0.2)
    generated_text = response['choices'][0]['text']
    return generated_text

# 2. Llama 모델 로드 및 텍스트 생성
def generate_text(prompt, max_new_tokens=256):
    response = llama(prompt, max_tokens=max_new_tokens, temperature=0.8)
    generated_text = response['choices'][0]['text']
    print(f"Generated Text:", generated_text)  # 생성된 텍스트 출력
    return generated_text

# 3. 생성된 텍스트에서 예상 형량 추출
def extract_expected_sentence(genesrated_text):
    match = re.search(r'(무죄|유예|벌금|징역\s*\d+\s*년?\s*\d*\s*월?|집행유예\s*\d+\s*년?\s*\d*\s*월?|몰수)', genesrated_text)
    print(f'match : ', match)
    if match:
        return match.group(0).strip()
    return "예상 형량을 추출할 수 없습니다."

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
        profession = summarize_generate_text(summarize_text(row['전문']))
        prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{row['양형의 이유']}\n{profession}\n\n"
        print(f'{i+1}번 전문 요약 : ', profession)

    prompt += f"현재 사건: {new_case_info}\n\n"
    prompt += "예상 형량:"
    return prompt

# 4. 생성된 텍스트에서 정보 추출 및 형식화
def generate_sentence(search_results, new_case_info, max_new_tokens=512):
    prompt = create_prompt(search_results, new_case_info)
    generated_text = generate_text(prompt, max_new_tokens)
    expected_sentence = extract_expected_sentence(generated_text)
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generated_text}")
    return generated_text
