from transformers import BertTokenizer, BertForSequenceClassification
import torch

class AIModule:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)  # from_safetensors 제거
        self.model.to(self.device)  # 모델을 디바이스로 이동

    def predict(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return "법률질문" if predictions.item() == 1 else "법률에 관련한 질문이 아닙니다. 법률에 관한 질문 입력하세요."