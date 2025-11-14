import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import CHROMA_DIR

# Load DB
db = Chroma(
    collection_name="instruct2ds",
    persist_directory=CHROMA_DIR,
    embedding_function=HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
)

# Lấy meta
metas = db._collection.get(limit=20000)["metadatas"]

embedded_sources = set([m["source"] for m in metas])

# Lấy toàn bộ file txt
all_txt = set(os.listdir("D:/Github/Final_DL/data/papers_text"))

# So sánh
not_embedded = [x for x in all_txt if x.replace(".txt","")+".txt" not in embedded_sources]

folder = r"D:/Github/Final_DL/data/papers_text"
txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

print("Số file txt:", len(txt_files))
print("Số file chưa được embed:", len(not_embedded))
print(not_embedded[:20])
