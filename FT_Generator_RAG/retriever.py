doc_url = '本地待检索的文档'
model_url = 'Embedding 模型的 本地路径'

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
loader = CSVLoader(file_path=doc_url)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
embeddings_model = HuggingFaceEmbeddings(model_url)
vector = FAISS.from_documents(documents, embeddings_model)
