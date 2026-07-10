from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("HF_TOKEN")

if not api_key:
    raise RuntimeError("HF_TOKEN not found")

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V4-Pro:fireworks-ai",
    api_key=api_key,
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=300,
)

response = llm.invoke("What is your knowledge cut off date?")

print(response.content)