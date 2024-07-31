import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load data and model
csv_path = r'C:/dev/python-model/merge_1_3_Deduplication.csv'
df = pd.read_csv(csv_path)
df['전문'] = df['전문'].fillna('')

search_model_path = r'C:/dev/python-model/KoSimCSE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(search_model_path)
model = AutoModel.from_pretrained(search_model_path)

# Load chunk embeddings
save_path = r'C:/dev/python-model/merge_1_3_Deduplication_chunk_embeddings_new.pt'
checkpoint = torch.load(save_path)
chunk_embeddings = checkpoint['chunk_embeddings'].to(device)
chunk_to_doc = checkpoint['chunk_to_doc']


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', '', text)
    text = text.lower()
    return text


def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return cls_embedding


def search_query(query, top_k=5):
    query = preprocess_text(query)
    query_embedding = encode_text(query).unsqueeze(0).to(device)

    query_embedding_cpu = query_embedding.cpu()
    chunk_embeddings_cpu = chunk_embeddings.cpu()

    similarities = cosine_similarity(query_embedding_cpu, chunk_embeddings_cpu).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    search_results = df.iloc[[chunk_to_doc[i] for i in top_indices]]
    #search_results['유사도'] = similarities[top_indices]
    search_results.loc[:, '유사도'] = similarities[top_indices]

    return search_results[['주문', '유사도']].to_dict(orient='records')
