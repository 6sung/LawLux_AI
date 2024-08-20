import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import torch
from kiwipiepy import Kiwi
import pickle

# Load data and model
csv_path = r'C:/dev/python-model/merge_1_6_Deduplication_cleaned_index.csv'
df = pd.read_csv(csv_path)
df['전문'] = df['전문'].fillna('')

search_model_path = r'C:/dev/python-model/KoSimCSE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(search_model_path)
model = AutoModel.from_pretrained(search_model_path)

# Load chunk embeddings
save_path = r'C:/dev/python-model/chunk_embeddings_1_6_512_128.pt'
checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
chunk_embeddings = checkpoint['chunk_embeddings'].to(device)
chunk_to_doc = checkpoint['chunk_to_doc']

# Load bm25 model
bm25_path = r'C:/dev/python-model/bm25_model_1_6.pkl'
with open(bm25_path, 'rb') as file:
    bm25_loaded = pickle.load(file)

# Load Reranker model
reranker_model_path = r"C:/dev/python-model/ko-reranker"
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path).to(device)
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)

def preprocess_text(text):
    kiwi = Kiwi()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', '', text)
    text = ' '.join([morph[0] for morph in kiwi.tokenize(text)])
    return text


def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return cls_embedding


# Reranker 함수
def rerank_documents(query, documents):
    rerank_scores = []

    for doc in documents:
        inputs = reranker_tokenizer(query, doc, return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            # logits의 크기를 확인하여 이진 분류인지 확인
            if outputs.logits.shape[1] == 1:
                score = outputs.logits.sigmoid().cpu().numpy()[0][0]  # 이진 분류의 경우
            else:
                score = outputs.logits.softmax(dim=1).cpu().numpy()[0][1]  # 다중 클래스의 경우
            rerank_scores.append(score)

    return rerank_scores

def search_query(query, top_k=5):
    query = preprocess_text(query)
    tokenized_query = query.split(" ")

    # BM25 점수 계산
    bm25_scores = bm25_loaded.get_scores(tokenized_query)

    # 상위 K개 인덱스 선택
    top_k = min(top_k, len(bm25_scores))  # top_k가 bm25_scores의 길이를 초과하지 않도록 조정
    top_indices = bm25_scores.argsort()[-top_k:][::-1]

    # Get top search results
    search_results = df.iloc[top_indices].copy()
    search_results['유사도'] = bm25_scores[top_indices]  # BM25 점수 추가

    search_results['전문'] = search_results['전문'].apply(lambda x: x[:160] + '...' if len(x) > 160 else x)

    rerank_scores= rerank_documents(query, search_results['전문'].tolist())
    search_results['리랭크'] = rerank_scores
    search_results = search_results.sort_values(by='리랭크', ascending=False)

    search_results['양형의 이유'] = search_results['양형의 이유'].fillna('')

    return search_results[['번호','사건번호', '주문', '유사도','전문', '양형의 이유']].to_dict(orient='records')
