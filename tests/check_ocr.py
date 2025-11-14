import fitz

doc = fitz.open("D:\Github\Final_DL\data\papers_raw\ACL_xxxx_a-calculus-for-semantic-composition-and-scoping.pdf")
for i, page in enumerate(doc):
    print(i, page.get_text("text")[:300])
