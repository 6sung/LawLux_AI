import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import pickle

# Load data and model
csv_path = r'C:/dev/python-model/merge_1_6_Deduplication_cleaned_index.csv'
df = pd.read_csv(csv_path)
df['전문'] = df['전문'].fillna('')
df['양형의 이유'] = df['양형의 이유'].fillna('')

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

kiwi = Kiwi()
def preprocess_text(text):
    text=re.sub(r'\s+',' ', text)
    text=re.sub(r'[^\w\s]', '', text)
    text=' '.join([morph[0] for morph in kiwi.tokenize(text)])
    return text

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return cls_embedding

def min_max_normalization(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)

def hybrid_cc(query, bm25_weight=0.26, cosine_weight=0.74, top_k=5, initial_top=10):
    # 쿼리 전처리
    query = preprocess_text(query)
    tokenized_query = query.split(" ")
    query_embedding = encode_text(query).unsqueeze(0).to(device)
    query_embedding_cpu = query_embedding.cpu()
    chunk_embeddings_cpu = chunk_embeddings.cpu()

    bm25_scores = bm25_loaded.get_scores(tokenized_query)

    similarities = cosine_similarity(query_embedding_cpu,
                                     chunk_embeddings_cpu).flatten()
    doc_similarities = [0] * (chunk_to_doc[-1] + 1)
    for i in range(len(chunk_to_doc)):
        if doc_similarities[chunk_to_doc[i]] < similarities[i]:
            doc_similarities[chunk_to_doc[i]] = similarities[i]

    normalized_bm25_scores = min_max_normalization(bm25_scores)
    normalized_cosine_similarities = min_max_normalization(doc_similarities)

    combined_scores = (bm25_weight * normalized_bm25_scores) + (cosine_weight * normalized_cosine_similarities)

    top_indices = combined_scores.argsort()[-initial_top:][::-1]

    search_results = df.iloc[top_indices].copy()
    search_results['bm25Score'] = bm25_scores[top_indices]
    search_results['유사도'] = similarities[top_indices]
    search_results['combinedScore'] = combined_scores[top_indices]

    search_results['전문'] = search_results['전문'].apply(lambda x: x[:120] + '...' if len(x) > 120 else x)

    seen_texts = set()
    filtered_results = []
    for _, row in search_results.iterrows():
        if row['전문'] not in seen_texts:
            seen_texts.add(row['전문'])
            filtered_results.append(row)

    filtered_results_df = pd.DataFrame(filtered_results)

    filtered_results_df = filtered_results_df[filtered_results_df['combinedScore'] >= 0.5]

    filtered_results_df = filtered_results_df.head(top_k)

    return filtered_results_df[['번호', '사건번호', '주문', '전문', '양형의 이유', 'bm25Score', '유사도', 'combinedScore']].to_dict(orient='records')
