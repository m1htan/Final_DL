from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

db = Chroma(
    collection_name="instruct2ds",
    persist_directory="D:\Github\Final_DL\db\chroma_instruct2ds",
    embedding_function=HuggingFaceBgeEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
)

print("Số lượng vector trong DB:", db._collection.count())
