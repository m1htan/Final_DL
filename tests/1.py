import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"D:\Github\Final_DL\data")
PDF_DIR = DATA_DIR / "papers_raw"
TEXT_DIR = DATA_DIR / "papers_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)
txt_files = []
for root, dirs, files in os.walk(TEXT_DIR):
    for f in files:
        if f.endswith(".txt"):
            txt_files.append(os.path.join(root, f))

print("Số file txt:", len(txt_files))
print(txt_files[:10])

print(open(txt_files[0], encoding="utf-8").read()[:500])
