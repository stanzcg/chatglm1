from langchain_community.document_loaders import PyPDFLoader
from langchain.evaluation.qa import QAGenerateChain
from langchain_community.llms import ChatGLM



endpoint_url='http://127.0.0.1:8000'
llm=ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    top_o=0.9
)
loader=PyPDFLoader('books/2022世界杯.pdf')
doc=loader.load()
example_gen_chain=QAGenerateChain.from_llm(llm=llm)
q=example_gen_chain.apply(
    [{'doc':doc[0]}]
)
print(q)