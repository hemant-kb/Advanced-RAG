from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    model="sonar",
    base_url="https://api.perplexity.ai",
    api_key=api_key,
    temperature=0.7,
    max_tokens=100,
)

response = llm.invoke("What is the capital of the USA?")
print(response.content)


