from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(directory='books'):
    loader=PyPDFDirectoryLoader(directory)
    doc=loader.load()
    text_spliter=CharacterTextSplitter(chunk_size=256,chunk_overlap=0)
    split_doc=text_spliter.split_documents(doc)
    return split_doc

def load_embedding_model(model_name = "sentence-transformers/all-mpnet-base-v2"):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf

def store_chroma(docs,embeddings,persist_directory='VectorStore'):
    db=Chroma.from_documents(docs,embeddings,persist_directory=persist_directory)
    db.persist()
    return db

