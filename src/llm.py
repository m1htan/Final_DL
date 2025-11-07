from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY, GEMINI_MODEL

def make_gemini():
    # Temperature thấp để lập kế hoạch ổn định; sẽ tinh chỉnh theo node sau.
    return ChatGoogleGenerativeAI(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
        temperature=0.2,
        convert_system_message_to_human=True,
    )
