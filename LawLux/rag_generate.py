import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd

# Load model and tokenizer
gen_model_name = r'C:/dev/python-model/llama-3-korean-bllossom-8b'
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, torch_dtype=torch.float16)
gen_model = gen_model.to('cpu')


def generate_sentence(search_results, new_case_info, max_length=1024, max_new_tokens=100):
    # prompt = '''
    #     현재 사건과 유사한 사건들 중 상위 5개의 사건의 판결 형량과 현재 사건과의 유사도를 참고하여 현재 사건에 대한 형량을 예측해 주세요. 결과는 형량만 출력하며, 설명은 필요 없습니다. 유사 사건의 출력도 필요 없고, 최종 예측 결과는 하나만 명확히 출력해 주세요. 형량은 다음 중 하나로 출력해 주세요:
    #     - 무죄 (유죄가 아닐 경우)
    #     - 유예
    #     - 벌금
    #     - 징역
    # '''
    #
    # if isinstance(search_results, list):
    #     search_results = pd.DataFrame(search_results)
    #
    # for i, row in search_results.iterrows():
    #     prompt += f"유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):\n{row['주문']}\n\n"
    #
    # prompt += f"현재 사건: {new_case_info}\n\n"
    # prompt += "예상 형량:"

    prompt = '서울시 날씨 알려줄래?'
    inputs = gen_tokenizer(prompt, return_tensors="pt").to('cpu')

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        pad_token_id=gen_tokenizer.eos_token_id
    )

    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    #expected_sentence = extract_expected_sentence(generated_text)

    #return expected_sentence

    return generated_text
def extract_expected_sentence(generated_text):
    match = re.search(r'예상 형량:\s*(.*)', generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "예상 형량을 추출할 수 없습니다."
