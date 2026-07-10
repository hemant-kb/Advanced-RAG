"""
LangGraph Chatbot Backend - Your AI-Powered PDF & Stock Analysis Assistant
This is the brain of the operation - handles conversations, documents, and stock data.
"""
from typing import Annotated, Iterator, List, Literal, TypedDict
import os
import shutil
import sqlite3
import gc
import time
import stat
from dotenv import load_dotenv

# LangChain essentials
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Graph framework
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages

# Our custom tools and scraper
from tools import (
    extract_text_from_pdf,
    chunk_text_into_documents,
    retrieve_relevant_context,
    search_stock_symbol,
    format_stock_data_for_display,
    format_stock_data_for_ai,
)
import scrap

# ============================================================================ #
# Configuration - Setting up our AI connections
# ============================================================================ #
load_dotenv()

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_key = os.getenv("GOOGLE_API_KEY")
#api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# ============================================================================ #
# Initialize AI Models
# ============================================================================ #
# llm = AzureChatOpenAI(
#     deployment_name=deployment_name,
#     azure_endpoint=azure_endpoint,
#     openai_api_key=api_key,
#     openai_api_version=api_version,
#     temperature=0.6,  # A bit creative but still grounded
#     streaming=True,
# )

# embeddings = AzureOpenAIEmbeddings(
#     deployment=embedding_deployment,
#     azure_endpoint=azure_endpoint,
#     openai_api_key=api_key,
#     openai_api_version=api_version,
# )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0.7,
    max_output_tokens=1000
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key,
    task_type="retrieval_document"
)

# model = HuggingFaceEndpoint(
#     repo_id="katanemo/Arch-Router-1.5B",
#     huggingfacehub_api_token=api_key,
#     temperature=0.3,
#     max_new_tokens=300,
#     task="text-generation"
# )

# # Wrap with ChatHuggingFace
# llm = ChatHuggingFace(llm=model)

# embeddings = HuggingFaceEndpointEmbeddings(
#     model="google/embeddinggemma-300m",
#     huggingfacehub_api_token=api_key
# )


# ============================================================================ #
# Storage - Where we keep everything organized
# ============================================================================ #
vector_stores = {}  # Document embeddings for each conversation
thread_filenames = {}  # Track which file belongs to which thread

# Database for persistence
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Create tables if they don't exist
conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT
    )
""")
conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_files (
        thread_id TEXT PRIMARY KEY,
        filename TEXT
    )
""")
conn.commit()


# ============================================================================ #
# Helper Functions
# ============================================================================ #

def normalize_text(text: str) -> str:
    """Clean up text - remove extra whitespace and blank lines"""
    if not text:
        return ""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def ensure_chroma_directory(thread_id: str) -> str:
    """Create a fresh directory for storing document embeddings"""
    base_dir = "./chroma_db"
    os.makedirs(base_dir, exist_ok=True)
    
    thread_dir = os.path.join(base_dir, thread_id)
    
    # Remove old directory if it exists (fresh start)
    if os.path.exists(thread_dir):
        shutil.rmtree(thread_dir)
    
    os.makedirs(thread_dir, exist_ok=True)
    return thread_dir


def force_remove_readonly(func, path, exc_info):
    """Helper to remove read-only files (Windows compatibility)"""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


# ============================================================================ #
# File Processing
# ============================================================================ #

def process_uploaded_file(file_path: str, file_type: str, thread_id: str, filename: str) -> int:
    """
    Process an uploaded PDF and make it searchable.
    Returns the number of chunks created.
    """
    file_type = file_type.lower()
    
    if file_type != "pdf":
        raise ValueError("Only PDF files are supported right now.")
    
    # Extract text from PDF
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        raise ValueError(
            "Couldn't extract text from this PDF. It might be scanned or empty. "
            "Try a text-based PDF instead."
        )
    
    # Clean up the text
    clean_text = normalize_text(raw_text)
    if not clean_text:
        raise ValueError("The PDF was processed but doesn't contain any readable text.")
    
    # Break into searchable chunks
    doc = Document(page_content=clean_text, metadata={"source": filename})
    chunks = chunk_text_into_documents(doc.page_content, thread_id, filename)
    
    if not chunks:
        raise ValueError("Couldn't create searchable chunks. The file might be too small.")
    
    # Store in vector database for fast retrieval
    persist_dir = ensure_chroma_directory(thread_id)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"thread_{thread_id}",
        persist_directory=persist_dir,
    )
    
    vector_stores[thread_id] = vectorstore
    thread_filenames[thread_id] = filename
    
    # Save to database
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO thread_files (thread_id, filename) VALUES (?, ?)",
        (thread_id, filename),
    )
    conn.commit()
    
    return len(chunks)


def get_thread_filename(thread_id: str) -> str:
    """Check if a thread has an uploaded file"""
    cur = conn.cursor()
    row = cur.execute(
        "SELECT filename FROM thread_files WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return row[0] if row else None


def delete_thread_files(thread_id: str) -> bool:
    """Clean up all files and data for a thread"""
    # Close vector store connection
    if thread_id in vector_stores:
        try:
            vectorstore = vector_stores[thread_id]
            if hasattr(vectorstore, '_client'):
                try:
                    vectorstore._client.clear_system_cache()
                except Exception:
                    pass
            if hasattr(vectorstore, '_collection'):
                collection_name = vectorstore._collection.name
                try:
                    vectorstore._client.delete_collection(collection_name)
                except Exception:
                    pass
        except Exception as e:
            print(f"Note: Had a small hiccup closing the vector store: {e}")
        finally:
            vector_stores.pop(thread_id, None)
    
    thread_filenames.pop(thread_id, None)
    gc.collect()
    time.sleep(0.5)
    
    # Remove the directory
    chroma_path = f"./chroma_db/{thread_id}"
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path, onerror=force_remove_readonly)
            return True
        except Exception as e:
            print(f"First attempt to delete failed: {e}. Trying again...")
            time.sleep(1)
            gc.collect()
            try:
                shutil.rmtree(chroma_path, onerror=force_remove_readonly)
                return True
            except Exception as e2:
                print(f"Couldn't delete the directory: {e2}")
                return False
    return True


# ============================================================================ #
# Graph State - What information flows through our conversation graph
# ============================================================================ #

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    retrieved_context: str
    needs_rag: bool
    needs_stock_analysis: bool
    stock_symbol: str
    stock_data: dict


# ============================================================================ #
# Graph Nodes - The decision-making parts of our system
# ============================================================================ #

def classifier_node(state: ChatState) -> dict:
    """
    The traffic controller - figures out what the user wants.
    Should we search a document? Analyze a stock? Or just chat?
    """
    thread_id = state.get("thread_id", "default")
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    has_document = thread_id in vector_stores
    document_filename = thread_filenames.get(thread_id, "a document")
    
    # If no document, just decide: Stock analysis or general chat?
    if not has_document:
        classification_prompt = [
            SystemMessage(content=(
                "You're helping me understand what the user wants. Pick ONE category:\n\n"
                "**STOCK_ANALYSIS** - They want info about Indian stocks (NSE/BSE)\n"
                "Examples: 'Analyze Titan', 'HCL quarterly results', 'Reliance PE ratio'\n\n"
                "**GENERAL_CHAT** - Everything else\n"
                "Examples: 'Write code', 'Explain ML', 'What's 2+2'\n\n"
                "Just output: STOCK_ANALYSIS or GENERAL_CHAT"
            )),
            HumanMessage(content=f"What does the user want?\n\n{last_message}")
        ]
        
        try:
            response = llm.invoke(classification_prompt)
            classification = response.content.strip().upper()
            
            if "STOCK" in classification:
                return {"needs_stock_analysis": True, "needs_rag": False}
            else:
                return {"needs_rag": False, "needs_stock_analysis": False}
        except Exception as e:
            print(f"Classification hiccup: {e}")
            return {"needs_rag": False, "needs_stock_analysis": False}
    
    # We have a document! Three-way decision: Stock, Document, or Chat?
    classification_prompt = [
        SystemMessage(content=(
            f"There's a document uploaded: '{document_filename}'. What does the user want?\n\n"
            "Pick ONE category:\n\n"
            "**STOCK_ANALYSIS** - Explicitly about Indian stock market analysis\n"
            "Examples: 'Analyze Titan stock', 'HCL quarterly results'\n\n"
            f"**DOCUMENT_QA** - Questions about '{document_filename}'\n"
            "Examples: 'How many leaves?', 'What does the policy say?', 'Summarize it'\n"
            "Important: Since a document IS available, if the question might be in it, choose this!\n\n"
            "**GENERAL_CHAT** - Clearly unrelated to the document or stocks\n"
            "Examples: 'Write factorial code', 'Tell me a joke', 'Explain quantum physics'\n\n"
            "Key rules:\n"
            f"- Document '{document_filename}' is available - prefer DOCUMENT_QA for questions!\n"
            "- Questions about policies, procedures, 'how many', 'what is' → DOCUMENT_QA\n"
            "- Only choose GENERAL_CHAT if it's obviously unrelated\n\n"
            "Output ONE word: STOCK_ANALYSIS, DOCUMENT_QA, or GENERAL_CHAT"
        )),
        HumanMessage(content=f"Classify:\n\n{last_message}")
    ]
    
    try:
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().upper()
        
        if "STOCK" in classification:
            return {"needs_stock_analysis": True, "needs_rag": False}
        elif "DOCUMENT" in classification:
            return {"needs_rag": True, "needs_stock_analysis": False}
        else:
            return {"needs_rag": False, "needs_stock_analysis": False}
    
    except Exception as e:
        print(f"Classification error: {e}")
        # Smart fallback: if document exists and it's a question, search it
        question_words = ['?', 'how', 'what', 'when', 'where', 'why', 'who', 'can', 'does', 'is', 'explain', 'tell me']
        if has_document and any(word in last_message.lower() for word in question_words):
            return {"needs_rag": True, "needs_stock_analysis": False}
        return {"needs_rag": False, "needs_stock_analysis": False}


def stock_symbol_extractor_node(state: ChatState) -> dict:
    """
    Figures out which stock the user is asking about.
    Uses AI + Screener.in search to find the right one.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Ask AI to extract the company name
    extraction_prompt = [
        SystemMessage(content=(
            "Extract the company name from what the user said.\n\n"
            "Examples:\n"
            "- 'Analyze TCS stock' → 'TCS'\n"
            "- 'Clean Science analysis' → 'Clean Science'\n"
            "- 'HCL Technologies data' → 'HCL Technologies'\n"
            "- 'Titan PE ratio' → 'Titan'\n\n"
            "Just output the company name, nothing else."
        )),
        HumanMessage(content=f"Extract company name:\n{last_message}")
    ]
    
    try:
        response = llm.invoke(extraction_prompt)
        company_name = response.content.strip()
    except Exception as e:
        print(f"Couldn't extract company name: {e}")
        return {"stock_symbol": "UNKNOWN"}
    
    # Search for the symbol
    symbol = search_stock_symbol(company_name, llm)
    return {"stock_symbol": symbol if symbol else "UNKNOWN"}


def stock_scraper_node(state: ChatState) -> dict:
    """Fetches all the financial data for the stock"""
    symbol = state.get("stock_symbol", "")
    
    if not symbol or symbol == "UNKNOWN":
        return {
            "stock_data": {
                "error": "I couldn't figure out which stock you meant. "
                "Could you give me a valid NSE symbol? (e.g., TITAN, RELIANCE, HCLTECH)"
            }
        }
    
    try:
        stock_data = scrap.scrape(symbol)
        return {"stock_data": stock_data}
    except ValueError as e:
        return {
            "stock_data": {
                "error": str(e),
                "symbol": symbol
            }
        }
    except Exception as e:
        return {
            "stock_data": {
                "error": f"Unexpected error getting data for {symbol}: {str(e)}",
                "symbol": symbol
            }
        }


def stock_analysis_response_node(state: ChatState) -> dict:
    """
    Generates the AI analysis of stock data.
    Shows the raw data first, then provides insights.
    """
    messages = state["messages"]
    stock_data = state.get("stock_data", {})
    symbol = state.get("stock_symbol", "")
    
    if "error" in stock_data:
        error_message = AIMessage(content=f"❌ {stock_data['error']}")
        return {"messages": [error_message]}
    
    # Format data for AI to analyze
    formatted_for_ai = format_stock_data_for_ai(stock_data)
    
    system_prompt = SystemMessage(content=(
        f"You're a financial analyst. Here's data for {symbol}. Provide:\n\n"
        "1. **Overview** - Brief company summary\n"
        "2. **Financial Performance** - Key metrics from results and P&L\n"
        "3. **Growth Analysis** - Analyze 1yr, 3yr, 5yr, 10yr growth trends\n"
        "4. **Key Ratios** - What the important ratios tell us\n"
        "5. **Balance Sheet** - Asset, liability, equity position\n"
        "6. **Cash Flow** - Operating, investing, financing flows\n"
        "7. **Shareholding** - Major shareholders and changes\n"
        "8. **Investment View** - Strengths, weaknesses, considerations\n\n"
        f"Stock Data:\n{formatted_for_ai}\n\n"
        "Use clear markdown formatting. Be factual and base everything on the data provided. "
        "Make the growth analysis insightful - explain what the trends mean for investors."
    ))
    
    analysis_messages = [system_prompt] + messages
    
    try:
        response = llm.invoke(analysis_messages)
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"Had trouble generating the analysis: {str(e)}")
        return {"messages": [error_msg]}


def rag_retrieval_node(state: ChatState) -> dict:
    """Searches the uploaded document for relevant information"""
    thread_id = state.get("thread_id", "default")
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    context = retrieve_relevant_context(thread_id, last_message, vector_stores, thread_filenames)
    return {"retrieved_context": context}


def rag_response_node(state: ChatState) -> dict:
    """Answers questions based on the document content"""
    messages = state["messages"]
    context = state.get("retrieved_context", "")
    thread_id = state.get("thread_id", "default")
    filename = thread_filenames.get(thread_id, "the document")
    
    system_msg = SystemMessage(content=(
        f"You're helping answer questions about '{filename}'.\n\n"
        "Guidelines:\n"
        "- Base your answer ONLY on the excerpts provided below\n"
        "- Quote specific details when relevant\n"
        "- If the excerpts don't have enough info, say so politely\n"
        "- Be specific and cite the document\n"
        "- Don't make things up or guess\n\n"
        f"**Document Excerpts:**\n{context}\n\n"
        "Now answer the user's question based ONLY on what's above."
    ))
    
    response = llm.invoke([system_msg] + messages)
    return {"messages": [response]}


def chat_node(state: ChatState) -> dict:
    """Handles general conversation - no documents or stocks needed"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================================ #
# Routing Logic - Decides where to send the conversation
# ============================================================================ #

def route_after_classifier(state: ChatState) -> Literal["stock_symbol_extractor", "rag_retrieval", "chat"]:
    """Send the conversation down the right path"""
    if state.get("needs_stock_analysis", False):
        return "stock_symbol_extractor"
    elif state.get("needs_rag", False):
        return "rag_retrieval"
    else:
        return "chat"


# ============================================================================ #
# Build the Conversation Graph
# ============================================================================ #

graph = StateGraph(ChatState)

# Add all our processing nodes
graph.add_node("classifier", classifier_node)
graph.add_node("stock_symbol_extractor", stock_symbol_extractor_node)
graph.add_node("stock_scraper", stock_scraper_node)
graph.add_node("stock_analysis_response", stock_analysis_response_node)
graph.add_node("rag_retrieval", rag_retrieval_node)
graph.add_node("rag_response", rag_response_node)
graph.add_node("chat", chat_node)

# Define the flow
graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier",
    route_after_classifier,
    {
        "stock_symbol_extractor": "stock_symbol_extractor",
        "rag_retrieval": "rag_retrieval",
        "chat": "chat"
    },
)

# Stock analysis path
graph.add_edge("stock_symbol_extractor", "stock_scraper")
graph.add_edge("stock_scraper", "stock_analysis_response")
graph.add_edge("stock_analysis_response", END)

# Document Q&A path
graph.add_edge("rag_retrieval", "rag_response")
graph.add_edge("rag_response", END)

# General chat path
graph.add_edge("chat", END)

# Compile it all together
chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================================ #
# Streaming Response with Progress Updates
# ============================================================================ #

def get_streaming_response_with_progress(user_message: str, thread_id: str):
    """
    Returns two iterators:
    1. Progress updates (what's happening behind the scenes)
    2. Final content (the actual response)
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get conversation history
    try:
        state = chatbot.get_state(config=config)
        previous_messages = state.values.get("messages", []) or []
    except Exception:
        previous_messages = []
    
    # Build the input with full history
    input_state = {
        "messages": list(previous_messages) + [HumanMessage(content=user_message)],
        "thread_id": thread_id,
    }
    
    # Progress updates
    def progress_gen():
        yield "🔍 Analyzing your query..."
    
    # Main content generator
    def content_gen():
        # Step 1: Figure out what the user wants
        classifier_result = classifier_node(input_state)
        input_state.update(classifier_result)
        
        # Stock analysis path
        if input_state.get("needs_stock_analysis"):
            # Extract symbol
            symbol_result = stock_symbol_extractor_node(input_state)
            input_state.update(symbol_result)
            
            symbol = input_state.get("stock_symbol", "UNKNOWN")
            
            if symbol == "UNKNOWN":
                yield "❌ I couldn't identify the stock symbol. Could you specify a valid NSE symbol like TITAN, RELIANCE, or HCLTECH?"
                return
            
            # Scrape data
            scraper_result = stock_scraper_node(input_state)
            input_state.update(scraper_result)
            
            stock_data = input_state.get("stock_data", {})
            
            if "error" in stock_data:
                yield f"❌ {stock_data['error']}"
                return
            
            # Show the raw data first
            full_data_display = format_stock_data_for_display(stock_data)
            yield full_data_display
            yield "\n\n---\n\n"
            
            # Now stream the AI analysis
            yield "# 🧠 AI Analysis\n\n"
            
            formatted_for_llm = format_stock_data_for_ai(stock_data)
            system_prompt = SystemMessage(content=(
                f"Analyze {symbol} and provide comprehensive insights. "
                "Cover performance, growth, ratios, balance sheet, cash flows, "
                "shareholding, and investment perspective.\n\n"
                f"Data:\n{formatted_for_llm}"
            ))
            
            messages_for_llm = [system_prompt] + input_state["messages"]
            for chunk in llm.stream(messages_for_llm):
                if chunk.content:
                    yield chunk.content
        
        # Document Q&A path
        elif input_state.get("needs_rag"):
            retrieval_result = rag_retrieval_node(input_state)
            input_state.update(retrieval_result)
            
            context = input_state.get("retrieved_context", "")
            system_msg = SystemMessage(content=(
                "Answer based on this document context:\n\n"
                f"{context}\n\n"
                "If the context doesn't have the answer, say so politely."
            ))
            
            messages_for_llm = [system_msg] + input_state["messages"]
            for chunk in llm.stream(messages_for_llm):
                if chunk.content:
                    yield chunk.content
        
        # General chat path
        else:
            for chunk in llm.stream(input_state["messages"]):
                if chunk.content:
                    yield chunk.content
    
    return progress_gen(), content_gen()


def get_streaming_response(user_message: str, thread_id: str):
    """Wrapper for compatibility with frontend"""
    progress_iter, content_iter = get_streaming_response_with_progress(user_message, thread_id)
    
    def combined_gen():
        for chunk in content_iter:
            yield chunk
    
    return combined_gen(), False


def get_progress_updates(user_message: str, thread_id: str):
    """
    Provides real-time progress updates as the system works.
    Shows users what's happening behind the scenes!
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = chatbot.get_state(config=config)
        previous_messages = state.values.get("messages", []) or []
    except Exception:
        previous_messages = []
    
    input_state = {
        "messages": list(previous_messages) + [HumanMessage(content=user_message)],
        "thread_id": thread_id,
    }
    
    yield {"step": "analyzing", "message": "🔍 Analyzing your query..."}
    
    classifier_result = classifier_node(input_state)
    input_state.update(classifier_result)
    
    if input_state.get("needs_stock_analysis"):
        yield {"step": "detected", "message": "📊 Stock analysis detected!"}
        
        yield {"step": "extracting", "message": "🔎 Finding the stock symbol..."}
        symbol_result = stock_symbol_extractor_node(input_state)
        input_state.update(symbol_result)
        
        symbol = input_state.get("stock_symbol", "UNKNOWN")
        
        if symbol == "UNKNOWN":
            yield {"step": "error", "message": "❌ Couldn't identify the symbol"}
            return
        
        yield {"step": "identified", "message": f"✅ Found it: **{symbol}**"}
        
        yield {"step": "fetching", "message": f"🌐 Fetching data from Screener.in..."}
        scraper_result = stock_scraper_node(input_state)
        input_state.update(scraper_result)
        
        stock_data = input_state.get("stock_data", {})
        
        if "error" in stock_data:
            yield {"step": "error", "message": "❌ Couldn't fetch the data"}
            return
        
        yield {"step": "fetched", "message": "✅ Data retrieved successfully!"}
        yield {"step": "analyzing_ai", "message": "🤖 Generating AI insights..."}
        yield {"step": "complete", "message": ""}
    
    elif input_state.get("needs_rag"):
        yield {"step": "searching", "message": "📄 Searching through the document..."}
        yield {"step": "generating", "message": "🤖 Crafting your answer..."}
        yield {"step": "complete", "message": ""}
    else:
        yield {"step": "processing", "message": "💬 Thinking about your message..."}
        yield {"step": "complete", "message": ""}


# ============================================================================ #
# UI Helper Functions
# ============================================================================ #

def retrieve_all_threads() -> list:
    """Get all conversation threads from the database"""
    cur = conn.cursor()
    
    # Try thread_titles first
    rows = cur.execute("SELECT thread_id FROM thread_titles ORDER BY rowid").fetchall()
    if rows:
        return [r[0] for r in rows]
    
    # Fallback to thread_files
    rows = cur.execute("SELECT thread_id FROM thread_files ORDER BY rowid").fetchall()
    seen = {}
    for r in rows:
        seen.setdefault(r[0], True)
    return list(seen.keys())


def delete_thread(thread_id: str):
    """Remove a conversation thread completely"""
    thread_id = str(thread_id)
    cur = conn.cursor()
    cur.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    cur.execute("DELETE FROM thread_files WHERE thread_id = ?", (thread_id,))
    conn.commit()
    delete_thread_files(thread_id)


def save_thread_title(thread_id: str, title: str):
    """Save or update a thread's title"""
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    conn.commit()


def get_thread_title(thread_id: str) -> str:
    """Get the title for a conversation thread"""
    cur = conn.cursor()
    row = cur.execute(
        "SELECT title FROM thread_titles WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return row[0] if row else thread_id


def generate_title_from_conversation(messages: list) -> str:
    """
    Auto-generate a catchy title from the first exchange.
    Saves users from having to name every conversation!
    """
    system_prompt = (
        "You're a title generator. Based on the first messages in this chat, "
        "create a short, clear title (max 6 words). "
        "Avoid punctuation like quotes or periods. Make it descriptive but concise."
    )
    
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join([f"{m.type}: {m.content}" for m in messages])},
    ]
    
    resp = llm.invoke(prompt_messages)
    return resp.content.strip()


def save_message_to_graph(thread_id: str, user_message: str, ai_message: str):
    """
    Save a conversation exchange to the graph state.
    This keeps the conversation history intact across sessions.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        current_state = chatbot.get_state(config=config)
        existing_messages = current_state.values.get("messages", []) if current_state.values else []
    except Exception:
        existing_messages = []
    
    new_messages = list(existing_messages) + [
        HumanMessage(content=user_message),
        AIMessage(content=ai_message),
    ]
    
    # Update the graph state with the new messages
    chatbot.update_state(
        config=config, 
        values={"messages": new_messages}
    )