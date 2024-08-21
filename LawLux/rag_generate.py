import re
import pandas as pd
from llama_cpp import Llama
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
# Load the generator model
generator_model_path = r'C:/dev/python-model/eeve/ggml-model-Q5_K_M.gguf'
llama = Llama(model_path=generator_model_path, n_ctx=4096)

#Load KoBART model and tokenizer - 요약
kobart_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')

# KoBART 요약 함수
def summarize_text(text, max_length=4096, min_length=300, length_penalty=2.0):
    inputs = kobart_tokenizer([text], max_length=4096, return_tensors='pt', truncation=True)
    summary_ids = kobart_model.generate(
        inputs['input_ids'],
        num_beams=8,
        max_length=max_length,
        min_length=min_length,  # 최소 요약 길이를 300으로 증가
        length_penalty=length_penalty,
        early_stopping=True
    )
    summary = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 후처리: 반복되는 문장 제거
    summary_lines = summary.split('. ')
    seen = set()
    unique_summary = []
    for line in summary_lines:
        if line not in seen:
            seen.add(line)
            unique_summary.append(line)
    summary = '. '.join(unique_summary)

    return summary

def reason_summarize_text(text, max_length=256):
    inputs = kobart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = kobart_model.generate(inputs['input_ids'], num_beams=4, max_length=max_length, early_stopping=True)
    summary = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_text(prompt, max_new_tokens=256):
    response = llama(prompt, max_tokens=max_new_tokens, temperature=0.1)
    generated_text = response['choices'][0]['text']
    print(f"Generated Text:", generated_text)  # 생성된 텍스트 출력
    return generated_text

def generate_text_stream(prompt, max_new_tokens=256):
    for token in llama(prompt, max_tokens=max_new_tokens, temperature=0.1, stream=True):
        yield token['choices'][0]['text']

def extract_expected_sentence(genesrated_text):
    match = re.search(r'(무죄|유예|벌금|징역\s*\d+\s*년?\s*\d*\s*월?|집행유예\s*\d+\s*년?\s*\d*\s*월?|몰수)', genesrated_text)
    print(f'match : ', match)
    if match:
        return match.group(0).strip()
    return "예상 형량을 추출할 수 없습니다."


def create_prompt(search_results, new_case_info):
    prompt = '''
            당신은 대한민국의 법률 전문가입니다.
            주어진 판례들만 고려하고, 외부 정보는 사용하지 마세요. 같은 말은 반복하지 마세요.

            이에 대한 응답은 다음과 같이 마크다운 리스트 형식으로(# 갯수 맞게 붙여주세요) 제공해주세요.

            #### 예상형량
            (유사판례를 참고하여 현재 사건에 대한 예상 형량을 예측해 주세요. 무죄, 징역 XX년/개월 집행유예 XX년/개월, 벌금 XX원, 징역 XX년/개월 중 하나만 나와야 합니다.)
            ##### 분석
            (유사 판례들의 판결 내용(전문)과 양형 이유를 참고하여 예상 형량 도출 이유를 설명)
            ##### 양형의 이유
            (유사 판례들의 양형 이유를 종합하여 현재 사건에 적용 가능한 양형 이유를 정리해주세요.)
            ##### 법률적 조언
            (피고인의 현재 사건에 대한 간단한 법률적 조언을 제공해주세요.)

            아래는 유사판례입니다:
        '''

    if isinstance(search_results, list):
        search_results = pd.DataFrame(search_results)

    for i, row in search_results.iterrows():
        # profession = summarize_generate_text(summarize_text(row['전문']))
        prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{row['양형의 이유']}\n\n"
        # print(f'{i+1}번 전문 요약 : ', profession)

    prompt += f"현재 사건: {new_case_info}\n\n"
    #prompt += "예상 형량:"
    return prompt

def generate_sentence_stream(search_results, new_case_info, max_new_tokens=650):
    prompt = create_prompt(search_results, new_case_info)
    print(f"Prompt: {prompt}")
    for token in generate_text_stream(prompt, max_new_tokens):
        yield token
        print(token, end='', flush=True)  # 디버깅을 위해 토큰 출력
    print()  # 줄바꿈

# 전체 텍스트를 생성하고 싶을 때 사용할 수 있는 함수
def generate_full_sentence(search_results, new_case_info, max_new_tokens=650):
    return ''.join(list(generate_sentence_stream(search_results, new_case_info, max_new_tokens)))
