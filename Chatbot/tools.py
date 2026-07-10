
"""
Tools Module - Handles RAG (Document Retrieval) and Stock Analysis
This is where the magic happens for document search and stock data analysis!
"""
import os
import re
import requests
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# We'll import these from back.py when needed
# from back import llm, embeddings, vector_stores, thread_filenames

# ============================================================================ #
# PDF Text Extraction - Getting text out of PDFs
# ============================================================================ #

def extract_with_pymupdf(path: str) -> str:
    """Try extracting PDF text using PyMuPDF (fast and reliable)"""
    try:
        import fitz
    except ImportError:
        return ""
    
    try:
        text_parts = []
        with fitz.open(path) as doc:
            # Handle password-protected PDFs (try empty password)
            if doc.is_encrypted:
                try:
                    doc.authenticate("")
                except Exception:
                    return ""
            
            # Extract text from each page
            for page in doc:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    text_parts.append(page_text)
        
        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def extract_with_pypdf(path: str) -> str:
    """Backup method using PyPDF (in case PyMuPDF fails)"""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    
    try:
        reader = PdfReader(path)
        
        # Handle encrypted PDFs
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return ""
        
        text_parts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception:
                continue
        
        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def extract_text_from_pdf(path: str) -> str:
    """
    Main PDF extraction function - tries multiple methods.
    Note: Scanned PDFs without embedded text won't work here.
    """
    for extractor in (extract_with_pymupdf, extract_with_pypdf):
        text = extractor(path)
        if text:
            return text
    return ""


# ============================================================================ #
# RAG (Retrieval-Augmented Generation) Tools
# ============================================================================ #

def chunk_text_into_documents(text: str, thread_id: str, source: str) -> List[Document]:
    """
    Breaks down large text into manageable, searchable chunks.
    Think of it like creating an index for a book - makes finding things easier!
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # About a page of text
        chunk_overlap=150,  # Keep some context between chunks
        separators=["\n\n", "\n", " ", ""],  # Try to split at natural boundaries
    )
    
    chunks = [chunk.strip() for chunk in splitter.split_text(text or "") if chunk and chunk.strip()]
    
    # Wrap each chunk in a Document with metadata
    return [
        Document(page_content=chunk, metadata={"thread_id": thread_id, "source": source}) 
        for chunk in chunks
    ]


def retrieve_relevant_context(thread_id: str, query: str, vector_stores: dict, thread_filenames: dict) -> str:
    """
    Searches through the uploaded document to find relevant information.
    Returns the most relevant excerpts that might answer the user's question.
    """
    if thread_id not in vector_stores:
        return ""
    
    # Search for the top 5 most relevant chunks
    retriever = vector_stores[thread_id].as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return "I couldn't find anything relevant in the document for that question."
    
    # Format the retrieved chunks nicely
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        if doc.page_content.strip():
            context_parts.append(f"[Excerpt {i}]\n{doc.page_content.strip()}")
    
    context = "\n\n".join(context_parts)
    filename = thread_filenames.get(thread_id, "the document")
    
    return f"Relevant excerpts from '{filename}':\n\n{context}"


# ============================================================================ #
# Stock Analysis Tools
# ============================================================================ #

def search_stock_symbol(company_name: str, llm) -> str:
    """
    Figures out the stock symbol from a company name.
    Uses Screener.in's search API and AI to pick the best match.
    """
    try:
        search_url = "https://www.screener.in/api/company/search/"    #https://www.screener.in/api/company/search/?q=berger
        params = {"q": company_name}
        
        response = requests.get(search_url, params=params, timeout=10)
        search_results = response.json()
        
        if not search_results:
            return "UNKNOWN"
        
        # If there's only one result, that's probably it!
        if len(search_results) == 1:
            url = search_results[0].get("url", "")
            return extract_symbol_from_url(url)
        
        # Multiple results? Let the AI pick the best one
        results_text = ""
        for i, result in enumerate(search_results[:5], 1):
            results_text += f"{i}. {result.get('name', 'N/A')} (URL: {result.get('url', 'N/A')})\n"
        
        matching_prompt = [
            SystemMessage(content=(
                f"Help me pick the right stock from these search results!\n\n"
                f"The user asked about: '{company_name}'\n\n"
                f"Here's what I found:\n{results_text}\n\n"
                "Which one is the best match? Consider:\n"
                "- Exact name matches or close variants\n"
                "- Well-known companies over obscure ones\n"
                "- Active companies over merged/delisted ones\n"
                "- Main company listings over subsidiaries\n\n"
                "Just tell me the number (1, 2, 3, etc.) of the best match."
            )),
            HumanMessage(content=f"Which result matches '{company_name}' best?")
        ]
        
        match_response = llm.invoke(matching_prompt)
        match_number = match_response.content.strip()
        
        # Extract the number from the response
        number_match = re.search(r'\d+', match_number)
        if number_match:
            index = int(number_match.group()) - 1
            if 0 <= index < len(search_results):
                url = search_results[index].get("url", "")
                return extract_symbol_from_url(url)
        
        # If we can't figure it out, just use the first result
        url = search_results[0].get("url", "")
        return extract_symbol_from_url(url)
        
    except Exception as e:
        print(f"Oops! Error searching for stock: {e}")
        return "UNKNOWN"


def extract_symbol_from_url(url: str) -> str:
    """
    Pulls out the stock symbol from a Screener.in URL.
    Example: /company/TCS/consolidated/ → TCS
    """
    # Look for alphabetic symbols first (these are usually NSE symbols)
    match = re.search(r'/company/([A-Z]+)', url)
    if match:
        return match.group(1)
    
    # Fallback to any alphanumeric code
    match = re.search(r'/company/([^/]+)', url)
    if match:
        symbol = match.group(1)
        if symbol.isalpha():
            return symbol.upper()
        return symbol
    
    return "UNKNOWN"


def format_stock_data_for_display(stock_data: Dict[str, Any]) -> str:
    """
    Formats raw stock data into a beautiful, readable markdown report.
    This is what the user sees - all the financial tables and data.
    """
    if not stock_data or "symbol" not in stock_data:
        return "No stock data available."
    
    parts = []
    symbol = stock_data.get('symbol', 'N/A')
    
    parts.append(f"# 📊 Stock Data for {symbol}\n")
    
    # Add each section if it has data
    sections = [
        ("quarterly", "📈 Quarterly Results"),
        ("profit_loss", "💰 Profit & Loss Statement"),
        ("growth_ranges", "📊 Growth Ranges"),
        ("balance_sheet", "🏦 Balance Sheet"),
        ("cash_flows", "💵 Cash Flow Statement"),
        ("ratios", "📊 Financial Ratios"),
    ]
    
    for key, title in sections:
        if stock_data.get(key) and stock_data[key][0]:
            parts.append(f"## {title}")
            parts.append(format_table_as_markdown(stock_data[key][0]))
            parts.append("")
    
    # Shareholding pattern (has nested structure)
    if stock_data.get("shareholding"):
        sh = stock_data["shareholding"]
        
        if sh.get("quarterly") and sh["quarterly"]:
            parts.append("## 👥 Shareholding Pattern (Quarterly)")
            parts.append(format_table_as_markdown(sh["quarterly"][0]))
            parts.append("")
        
        if sh.get("yearly") and sh["yearly"]:
            parts.append("## 👥 Shareholding Pattern (Yearly)")
            parts.append(format_table_as_markdown(sh["yearly"][0]))
            parts.append("")
    
    return "\n".join(parts)


def format_stock_data_for_ai(stock_data: Dict[str, Any]) -> str:
    """
    Formats stock data in a compact text format that's easy for the AI to analyze.
    This is the behind-the-scenes version that feeds into the AI's analysis.
    """
    if not stock_data or "symbol" not in stock_data:
        return "No stock data available."
    
    parts = [f"Stock Symbol: {stock_data.get('symbol', 'N/A')}\n"]
    
    # Compact text format for AI processing
    sections = [
        ("quarterly", "QUARTERLY RESULTS"),
        ("profit_loss", "PROFIT & LOSS"),
        ("growth_ranges", "GROWTH RANGES"),
        ("ratios", "KEY RATIOS"),
        ("balance_sheet", "BALANCE SHEET"),
        ("cash_flows", "CASH FLOWS"),
    ]
    
    for key, title in sections:
        if stock_data.get(key) and stock_data[key][0]:
            parts.append(f"\n=== {title} ===")
            table = stock_data[key][0]
            if table.get("columns") and table.get("rows"):
                parts.append("Columns: " + " | ".join(table["columns"]))
                for row in table["rows"]:
                    parts.append(" | ".join(str(cell) for cell in row))
    
    # Add shareholding data
    if stock_data.get("shareholding"):
        sh = stock_data["shareholding"]
        if sh.get("quarterly") and sh["quarterly"]:
            parts.append("\n=== SHAREHOLDING PATTERN (Quarterly) ===")
            q_sh = sh["quarterly"][0]
            if q_sh.get("columns") and q_sh.get("rows"):
                parts.append("Columns: " + " | ".join(q_sh["columns"]))
                for row in q_sh["rows"]:
                    parts.append(" | ".join(str(cell) for cell in row))
    
    return "\n".join(parts)


def format_table_as_markdown(table_data: Dict[str, Any], max_rows: int = None) -> str:
    """
    Converts table data into nice markdown format.
    Makes those financial tables look good!
    """
    if not table_data or not table_data.get("columns") or not table_data.get("rows"):
        return "No data available"
    
    columns = table_data["columns"]
    rows = table_data["rows"]
    
    if max_rows:
        rows = rows[:max_rows]
    
    # Create markdown table
    header = "| " + " | ".join(str(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    
    table_rows = []
    for row in rows:
        table_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join([header, separator] + table_rows)