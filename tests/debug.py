from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CHROMA_DIR, EMBEDDING_MODEL

db = Chroma(
    collection_name="instruct2ds",
    persist_directory=CHROMA_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)

all = db.get()
print(len(all["documents"]))
print(all["metadatas"][:5])
