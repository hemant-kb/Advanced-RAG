from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Get HuggingFace API token
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Initialize HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    huggingfacehub_api_token=api_key,
    temperature=0.7,
    max_new_tokens=128,
    task="text-generation"
)

# Wrap with ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# Use with messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is credit fraud")
]

response = chat_model.invoke(messages)
print(response.content)

embeddings = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=api_key
)


# documents = [
#     "Python is a high-level programming language.",
#     "Machine learning is a subset of artificial intelligence.",
#     "Natural language processing helps computers understand human language."
# ]

# # Use same way as above
# doc_embeddings = embeddings.embed_documents(documents)
# query_embedding = embeddings.embed_query("What is Python?")

# print(query_embedding)


