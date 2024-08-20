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
    response = llama(prompt, max_tokens=max_new_tokens, temperature=0.8)
    generated_text = response['choices'][0]['text']
    print(f"Generated Text:", generated_text)  # 생성된 텍스트 출력
    return generated_text



def extract_expected_sentence(genesrated_text):
    match = re.search(r'(무죄|유예|벌금|징역\s*\d+\s*년?\s*\d*\s*월?|집행유예\s*\d+\s*년?\s*\d*\s*월?|몰수)', genesrated_text)
    print(f'match : ', match)
    if match:
        return match.group(0).strip()
    return "예상 형량을 추출할 수 없습니다."


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

    # prompt = '''
    # 아래 주어진 현재 사건과 유사도가 높은 사건들을 참고하여 현재 사건에 대한 형량을 예측해 주세요. 아래 주어진 사례들만 고려하세요. 외부 리소스는 사용하지 마세요.
    # 유사도는 현재 사건과 유사한 정도, 주문에는 유사한 사건들의 형량, 전문에는 유사한 사건들의 당시 상황이 주어집니다.
    # 예상 형량은 다음 중 하나로 출력해 주세요(사회봉사, XX강의 수강, 취업 제한 등 내용은 절대 예상 형량에 포함시키지 마세요.):
    # - 무죄 (유죄가 아닐 경우)
    # - 유예일 경우 형식 >> 징역 XX년(개월) 집행유예 XX년(개월)
    # - 벌금일 경우 형식 >> XX원
    # - 징역일 경우 형식 >> 징역 XX년(개월)
    # 줄바꿈 한번하고나서,
    # 사용자의 현재 상태에 대한 간단한 법률적 조언도 출력해주세요.
    # '''

    if isinstance(search_results, list):
        search_results = pd.DataFrame(search_results)

    for i, row in search_results.iterrows():
        # profession = summarize_generate_text(summarize_text(row['전문']))
        prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{row['양형의 이유']}\n\n"
        # print(f'{i+1}번 전문 요약 : ', profession)

    prompt += f"현재 사건: {new_case_info}\n\n"
    prompt += "예상 형량:"
    return prompt

    # print(len(row['전문']))
        # print(len(row['양형의 이유']))
        # if(len(row['양형의 이유'])>200):
        #     reason = reason_summarize_text(row['양형의 이유'])
        # elif len(row['양형의 이유']) > 0:
        #     reason = row['양형의 이유']
        # else : reason = '양형의 이유 없음.'
        # prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n양형의 이유 : {reason}\n\n"
        #
        # if len(row['전문']) > 900:
        #     print(row['전문'])
        #     summarize = summarize_text(row['전문'])
        #     print(summarize)
        #     prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{summarize}\n{reason}\n\n"
        # else:
        #     print(row['전문'])
        #     prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{row['전문']}\n{reason}\n\n"
    prompt += f"현재 사건: {new_case_info}\n\n"
    prompt += "예상 형량:"

    return prompt


# def extract_expected_sentence(generated_text):
#     match = re.search(r'(징역\s*\d+\s*년\s*\d+\s*월|형량\s*:\s*[\s\S]+)', generated_text, re.DOTALL)
#     print(f'match : ', match)
#     if match:
#         return match.group(1).strip()
#     return "예상 형량을 추출할 수 없습니다."
#
#
# def create_prompt(search_results, new_case_info):
#     prompt = ""
#     if isinstance(search_results, list):
#         search_results = pd.DataFrame(search_results)
#
#     for i, row in search_results.iterrows():
#         prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n\n"
#
#     prompt += f"현재 사건: {new_case_info}\n\n"
#     prompt += "예상 형량:"
#
#     return prompt


def generate_sentence(search_results, new_case_info, max_new_tokens=512):
    prompt = create_prompt(search_results, new_case_info)
    generated_text = generate_text(prompt, max_new_tokens)
    expected_sentence = extract_expected_sentence(generated_text)
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generated_text}")
    #print(f"Extracted Sentence: {expected_sentence}")
    #return expected_sentence
    return generated_text

# Example usage

#     if isinstance(search_results, list):
#         search_results = pd.DataFrame(search_results)
#
#     for i, row in search_results.iterrows():
#         prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n\n"
#
#     prompt += f"현재 사건: {new_case_info}\n\n"
#     prompt += "예상 형량:"
#
#     inputs = gen_tokenizer(prompt, return_tensors="pt").to('cpu')
#
#     outputs = gen_model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=0.8,
#         pad_token_id=gen_tokenizer.eos_token_id
#     )
#
#     generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     #expected_sentence = extract_expected_sentence(generated_text)
#
#     #return expected_sentence
#
#     return generated_text
# def extract_expected_sentence(generated_text):
#     match = re.search(r'예상 형량:\s*(.*)', generated_text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return "예상 형량을 추출할 수 없습니다."
