import faiss
from fastapi import FastAPI
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

app = FastAPI(
    title="AASC Sentence Search",
    description="Find the most similar sentences in ACL Anthology (from 2010 to 2018) ..."
)

# embeddings preparation
npz = np.load('./data/AASC_embeddings.npz')
embeddings = npz["arr_0"]
D = embeddings.shape[1]
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(D)
index.add(embeddings)

# embeddings info preparation
df = pd.read_csv('./data/AASC_embeddings.tsv', header=None, sep="\t")

# model preparation
model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


default_sentence = "Named Entity Recognition (NER) is a fundamental task in the fields of natural language processing and information extraction."
default_top_k = 10

@app.post("/search")
async def find_the_most_similar_sentences(text: str = default_sentence, top_k: int = default_top_k):
    batch = tokenizer([text], padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**batch)
        embedding = outputs.last_hidden_state[0, 0, :]
        embedding = embedding.cpu().detach().numpy().copy().astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding, ord=2)

        dists, ids = index.search(x=np.array([embedding]), k=top_k)

        res = []
        for i in ids[0]:
            res.append({"id": df.at[i, 1][1:-4], "text": df.at[i, 0][1:-1]})

        return res
