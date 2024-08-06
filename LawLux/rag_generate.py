import re
import pandas as pd
from llama_cpp import Llama

# Load the generator model
generator_model_path = r'C:/dev/python-model/ggml-model-Q5_K_M.gguf'
llama = Llama(model_path=generator_model_path)


# 1. 프롬프트 생성
def create_prompt_parts(search_results, new_case_info):
    try:
        prompt_parts = []
        if isinstance(search_results, list):
            search_results = pd.DataFrame(search_results)

        for i, row in search_results.iterrows():
            if i >= 3:
                break
            # 유사 사건 3개만 포함
            part = f"\n\n유사 사건 {i + 1} (유사도: {row['유사도']:.2f}):"
            part += f"\n전문: {row['전문']}"
            part += f"\n양형의 이유: {row['양형의 이유']}"
            part += f"\n주문: {row['주문']}"
            prompt_parts.append(part)

        # 현재 사건 정보 추가
        current_case_part = f"\n\n현재 사건:\n{new_case_info}\n\n"
        prompt_parts.append(current_case_part)

        return prompt_parts
    except Exception as e:
        print(f"Error in create_prompt_parts: {str(e)}")
        return str(e)


# 2. Llama 모델 로드 및 텍스트 생성
def generate_text(prompt, max_new_tokens=100):
    try:
        response = llama(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            stop=['</s>'],
            echo=False
        )
        generated_text = response['choices'][0]['text']
        print(f"Generated Text: {generated_text}")  # 생성된 텍스트 출력
        return generated_text
    except Exception as e:
        print(f"Error in generate_text: {str(e)}")
        return str(e)


# 3. 생성된 텍스트에서 예상 형량 추출
def extract_expected_sentence(generated_text):
    try:
        # 예상 형량 추출
        sentence_match = re.search(r'(무죄|유예|벌금|징역\s*\d+\s*년?|집행유예|몰수)', generated_text)
        if sentence_match:
            sentence = sentence_match.group(0).strip()
        else:
            sentence = "예상 형량을 추출할 수 없습니다."

        # 양형의 이유 추출
        reasons_match = re.search(r'양형의 이유[:\s]*(1\.\s[\s\S]*?)(?:2\.\s|분석[:\s]*|$)', generated_text, re.DOTALL)
        if reasons_match:
            reasons = reasons_match.group(1).strip()
        else:
            reasons = "양형의 이유를 추출할 수 없습니다."

        # 분석 추출
        analysis_match = re.search(r'분석[:\s]*(1\.\s[\s\S]*?)(?:참조 판례[:\s]*|$)', generated_text, re.DOTALL)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        else:
            analysis = "분석을 추출할 수 없습니다."

        # 유사 사건 추출
        precedents_match = re.search(r'참조 판례[:\s]*(1\.\s[\s\S]*)', generated_text, re.DOTALL)
        if precedents_match:
            precedents = precedents_match.group(1).strip()
        else:
            precedents = "참조 판례를 추출할 수 없습니다."

        return {
            'sentence': sentence,
            'reasons': reasons,
            'analysis': analysis,
            'precedents': precedents
        }
    except Exception as e:
        print(f"Error in extract_expected_sentence: {str(e)}")
        return {
            'sentence': "예상 형량을 추출할 수 없습니다.",
            'reasons': "양형의 이유를 추출할 수 없습니다.",
            'analysis': "분석을 추출할 수 없습니다.",
            'precedents': "참조 판례를 추출할 수 없습니다."
        }


# 4. 생성된 텍스트에서 정보 추출 및 형식화
def generate_sentence(search_results, new_case_info, max_new_tokens=100):
    prompt_parts = create_prompt_parts(search_results, new_case_info)

    full_text = ""
    for part in prompt_parts:
        generated_text = generate_text(part, max_new_tokens)
        full_text += generated_text

    extracted_info = extract_expected_sentence(full_text)

    # 원하는 형식으로 출력
    formatted_output = f"""
        예상 형량: {extracted_info['sentence']}

        양형의 이유:
        {extracted_info['reasons']}

        분석:
        {extracted_info['analysis']}

        참조 판례:
        {extracted_info['precedents']}
    """
    print(formatted_output)
    return extracted_info