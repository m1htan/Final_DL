from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5:7b")
print(llm.invoke("Xin chào, bạn là ai?"))
