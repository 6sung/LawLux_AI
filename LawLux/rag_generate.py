import re
import pandas as pd
from llama_cpp import Llama

# Load the generator model
generator_model_path = r'C:/dev/python-model/eeve/ggml-model-Q5_K_M.gguf'
llama = Llama(model_path=generator_model_path)


def generate_text(prompt, max_new_tokens=150):
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
    현재 사건과 유사한 사건들 중 상위 5개의 사건의 판결 형량과 현재 사건과의 유사도를 참고하여 현재 사건에 대한 형량을 예측해 주세요. 
    결과는 형량만 출력하며, 설명은 필요 없습니다. 유사 사건의 출력도 필요 없고, 최종 예측 결과는 하나만 명확히 출력해 주세요. 
    형량은 다음 중 하나로 출력해 주세요:
    - 무죄 (유죄가 아닐 경우)
    - 유예
    - 벌금
    - 징역
    '''

    if isinstance(search_results, list):
        search_results = pd.DataFrame(search_results)

    for i, row in search_results.iterrows():
        prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n\n"

    prompt += f"현재 사건: {new_case_info}\n\n"
    gen_prompt = "예상 형량:"

    return gen_prompt


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


def generate_sentence(search_results, new_case_info, max_new_tokens=100):
    prompt = create_prompt(search_results, new_case_info)
    generated_text = generate_text(prompt, max_new_tokens)
    expected_sentence = extract_expected_sentence(generated_text)
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generated_text}")
    print(f"Extracted Sentence: {expected_sentence}")
    return expected_sentence


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
