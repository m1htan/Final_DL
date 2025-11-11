from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL

db = Chroma(
    collection_name="instruct2ds",
    persist_directory="D:\Github\Final_DL\db\chroma_instruct2ds",
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)

print("Số lượng vector trong DB:", db._collection.count())

