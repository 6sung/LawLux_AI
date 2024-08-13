import re
import pandas as pd
from llama_cpp import Llama
# from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# Load the generator model
#generator_model_path = r'D:/dev/python-model/qwen/qwen2-7b-instruct-q5_k_m.gguf' # qwen
generator_model_path = r'C:/dev/python-model/eeve/ggml-model-Q5_K_M.gguf' # eeve
#generator_model_path = r'D:/dev/python-model/llama3_luxia/Ko-Llama3-Luxia-8B.Q5_K_M.gguf' # llama3_luxia
llama = Llama(model_path=generator_model_path, n_ctx=4096)

# # Load KoBART model and tokenizer - 요약
# kobart_model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
# kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
#
# # KoBART 요약 함수
# def summarize_text(text, max_length=256):
#     inputs = kobart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
#     summary_ids = kobart_model.generate(inputs['input_ids'], num_beams=4, max_length=max_length, early_stopping=True)
#     summary = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# 2. Llama 모델 로드 및 텍스트 생성
def generate_text(prompt, max_new_tokens=256):
    response = llama(prompt=prompt, max_tokens=max_new_tokens, temperature=0.8)
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
        아래 주어진 현재 사건과 유사도가 높은 사건들을 참고하여 현재 사건에 대한 형량을 예측해 주세요. 아래 주어진 사례들만 고려하세요. 외부 리소스는 사용하지 마세요.
        유사도는 현재 사건과 유사한 정도, 주문에는 유사한 사건들의 형량, 전문에는 유사한 사건들의 당시 상황이 주어집니다.
        내용을 출력할 때 반복되는 말은 하지 마세요.
        명확하게 말하세요.

        예상 형량은 다음 중 하나로 출력해 주세요(XX강의 수강, 취업 제한 등 내용은 출력 필요 없습니다.) :
        - 무죄 (유죄가 아닐 경우)
        - 유예일 경우 형식 (징역 XX년/개월 집행유예 XX년/개월)
        - 벌금일 경우 형식 XX원
        - 징역일 경우 형식 (징역 XX년/개월)

        예상 형량을 도출할 때 다음 항목을 반드시 포함하세요:
        1. **예측 형량**: 유사 사건들의 주문 내용을 참고하여 예상 형량을 명확하게 제시하세요.
        2. **분석**: 예상 형량을 도출한 이유를 설명하세요. 유사 사건들의 전문과 양형의 이유를 바탕으로, 형량에 영향을 미친 요소들을 구체적으로 분석하세요.
        3. **양형의 이유**: 유사 사건들에서 언급된 양형의 이유를 종합하여, 현재 사건에 적용될 수 있는 양형의 이유를 정리하세요.
        4. **법률적 조언**: 사용자의 현재 상태에 대한 간단한 법률적 조언을 제공하세요. 예를 들어, 형량을 줄이기 위해 고려해야 할 요소나 법적 조치를 제안하세요.

        아래는 유사한 사건들에 대한 정보입니다:
        '''

    if isinstance(search_results, list):
        search_results = pd.DataFrame(search_results)

    for i, row in search_results.iterrows():
        prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n{row['전문']}\n"
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

# 4. 생성된 텍스트에서 정보 추출 및 형식화
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
if __name__ == "__main__":
    search_results = [
        {"번호": 1, "주문": "주문 내용 예시 1", "유사도": 0.95, "전문": "전문 내용 예시 1..."},
        {"번호": 2, "주문": "주문 내용 예시 2", "유사도": 0.90, "전문" : "내용 예시 2..."}
    ]
    new_case_info = "현재 사건의 정보"
    gen_sentence = generate_sentence(search_results, new_case_info)
    print(gen_sentence)


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