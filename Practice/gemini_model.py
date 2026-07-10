from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",   # ✅ Updated: stable GA model name
    google_api_key=api_key,
    temperature=0.7,
    max_output_tokens=300
)

response = llm.invoke("What is the capital of the USA?")
print(response.content)