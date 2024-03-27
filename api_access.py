from QA import load_documents,load_embedding_model,store_chroma
import os
import requests
from langchain_community.vectorstores import Chroma
def chat(prompt,history):
    resp=requests.post(
        url='http://127.0.0.1:8000',
        json={'prompt':prompt,'history':history},
        headers={'Content-Type':'application/json;charset=utf-8'}
    )
    return resp.json()['response'],resp.json()['history']

embeddings=load_embedding_model()
if not os.path.exists('VectorStore'):
    documents=load_documents()
    db=store_chroma(documents,embeddings)
else:
    db=Chroma(persist_directory='VectorStore',embedding_function=embeddings)

or_history=[]
while True:
    message=input('Question:')
    similar_docs = db.similarity_search(message, k=4)
    prompt = '基于下面给出的资料回答问题，如果资料不充足就回复不知道。下面是资料：\n'
    for idx, doc in enumerate(similar_docs):
        prompt += f'{idx + 1}. {doc.page_content}\n'
    prompt += f'下面是问题: {message}'
    print(prompt)
    response, history = chat(prompt, or_history)
    or_history+=history
    print('Answer:',response)