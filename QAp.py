from QA import load_documents,load_embedding_model,store_chroma
import gradio as gr
import os
import requests
import time
from langchain_community.vectorstores import Chroma

def chat(prompt,history):
    payload={
        'prompt':prompt,
        'history': [] if not history else history,
    }
    headers={'Content-Type':'application/json'}
    resp=requests.post(
        url='http://127.0.0.1:8000',
        json=payload,
        headers=headers,
    ).json()
    return resp['response'],resp['history']

embeddings=load_embedding_model()
if not os.path.exists('VectorStore'):
    documents=load_documents()
    db=store_chroma(documents,embeddings)
else:
    db=Chroma(persist_directory='VectorStore',embedding_function=embeddings)

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    directory=os.path.dirname(file.name)
    doc=load_documents(directory)
    store_chroma(doc,embeddings)
    history = history + [((file.name,), None)]
    return history

def bot(history):
    message=history[-1][0]
    if isinstance(message,tuple):
        response='Success!'
    else:
        similar_docs=db.similarity_search(message,k=4)
        prompt='基于下面给出的资料回答问题，如果资料不充足就回复不知道。下面是资料：\n'
        for idx,doc in enumerate(similar_docs):
            prompt+=f'{idx+1}. {doc.page_content}\n'
        prompt += f'下面是问题: {message}'
        response, _ = chat(prompt,history)
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=['pdf'])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)
demo.queue()
demo.launch(inbrowser=True)
