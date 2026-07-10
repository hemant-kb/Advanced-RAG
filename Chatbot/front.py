"""
LangGraph Chatbot Frontend - Your Interactive AI Assistant Interface
Beautiful, intuitive chat interface with PDF upload and stock analysis!
"""
import os
import tempfile
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from back import (
    chatbot,
    delete_thread,
    generate_title_from_conversation,
    get_streaming_response,
    get_progress_updates,
    get_thread_filename,
    get_thread_title,
    process_uploaded_file,
    retrieve_all_threads,
    save_message_to_graph,
    save_thread_title,
)

# ============================================================================ #
# Page Configuration - Make it look good!
# ============================================================================ #
st.set_page_config(
    page_title="LangGraph Chatbot", 
    page_icon="🤖", 
    layout="wide"
)

# Custom CSS for a modern, polished look
st.markdown(
    """
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"][aria-expanded="true"] { 
        width: 30px; 
        min-width: 380px; 
        max-width: 380px; 
    }
    [data-testid="stSidebar"] { 
        transition: all 0.3s ease-in-out; 
    }
    [data-testid="stSidebarContent"] { 
        padding-top: 1rem; 
        overflow-y: auto; 
    }
    
    /* Document indicator - dark elegant theme */
    .file-indicator { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0b1220 100%); 
        border-left: 4px solid #22d3ee; 
        box-shadow: 0 6px 20px rgba(2, 6, 23, 0.35); 
        border-radius: 14px; 
        padding: 14px 16px; 
        margin: 14px 0 18px 0; 
        color: #e2e8f0; 
        line-height: 1.4; 
    }
    .file-indicator strong { 
        color: #ffffff; 
        font-weight: 600; 
    }
    .file-indicator small { 
        color: #cbd5e1; 
        font-size: 0.9rem; 
    }
    .file-indicator .fi-row { 
        display: flex; 
        align-items: center; 
        gap: 10px; 
    }
    
    /* Stock analysis indicator - vibrant green theme */
    .stock-indicator { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f0f1e 100%); 
        border-left: 4px solid #4ade80; 
        box-shadow: 0 6px 20px rgba(2, 6, 23, 0.35); 
        border-radius: 14px; 
        padding: 14px 16px; 
        margin: 14px 0 18px 0; 
        color: #e2e8f0; 
        line-height: 1.4; 
    }
    .stock-indicator strong { 
        color: #ffffff; 
        font-weight: 600; 
    }
    .stock-indicator small { 
        color: #cbd5e1; 
        font-size: 0.9rem; 
    }
    
    /* Spacing improvements */
    [data-testid="stPopover"] { 
        margin-bottom: 0; 
    }
    div.block-container { 
        padding-bottom: 0.5rem; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================ #
# Helper Functions
# ============================================================================ #

def generate_thread_id() -> str:
    """Create a unique ID for a new conversation"""
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    """Add a new thread to our session state"""
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    """Load the complete conversation history for a thread"""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", []) or []
    except Exception:
        messages = []
    
    # Convert to simple format for display
    chat_history = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = getattr(msg, "content", "")
        if content:
            chat_history.append({"role": role, "content": content})
    
    return chat_history


def switch_thread(thread_id: str):
    """Switch to a different conversation"""
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)
    st.rerun()


def reset_chat():
    """Start a fresh conversation"""
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    add_thread(new_id)
    save_thread_title(new_id, "New Conversation")
    st.rerun()


# ============================================================================ #
# Initialize Session State
# ============================================================================ #

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

active = st.session_state.get("thread_id")

# Make sure we have an active thread
if st.session_state["chat_threads"]:
    if active not in st.session_state["chat_threads"]:
        st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
else:
    # No threads? Create the first one!
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["chat_threads"] = [new_id]
    save_thread_title(new_id, "New Conversation")

# Load conversation history
st.session_state.setdefault(
    "message_history", 
    load_conversation(st.session_state["thread_id"])
)

# Track deletion state
if "pending_delete_id" not in st.session_state:
    st.session_state["pending_delete_id"] = None
if "pending_delete_title" not in st.session_state:
    st.session_state["pending_delete_title"] = ""

# Track editing state
if "editing_thread_id" not in st.session_state:
    st.session_state["editing_thread_id"] = None

# Track uploaded files
if "uploaded_once" not in st.session_state:
    st.session_state["uploaded_once"] = {}


# ============================================================================ #
# Sidebar - Conversation Management
# ============================================================================ #

st.sidebar.title("🤖 LangGraph Chatbot")

# Show delete confirmation if needed
if st.session_state["pending_delete_id"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Confirm Deletion")
    st.sidebar.warning(
        f"Delete **{st.session_state['pending_delete_title']}**?", 
        icon="⚠️"
    )
    
    dc1, dc2 = st.sidebar.columns(2)
    
    with dc1:
        if st.button(
            "Yes, delete", 
            key="confirm_delete", 
            type="primary", 
            use_container_width=True
        ):
            tid = st.session_state["pending_delete_id"]
            delete_thread(tid)
            
            if tid in st.session_state["chat_threads"]:
                st.session_state["chat_threads"].remove(tid)
            
            # Switch to another thread or create new one
            if st.session_state.get("thread_id") == tid:
                if st.session_state["chat_threads"]:
                    st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
                    st.session_state["message_history"] = load_conversation(
                        st.session_state["thread_id"]
                    )
                else:
                    new_id = generate_thread_id()
                    st.session_state["thread_id"] = new_id
                    st.session_state["chat_threads"] = [new_id]
                    save_thread_title(new_id, "New Conversation")
                    st.session_state["message_history"] = []
            
            st.session_state["pending_delete_id"] = None
            st.session_state["pending_delete_title"] = ""
            st.rerun()
    
    with dc2:
        if st.button("Cancel", key="cancel_delete", use_container_width=True):
            st.session_state["pending_delete_id"] = None
            st.session_state["pending_delete_title"] = ""
            st.rerun()

# New chat button
if not st.session_state["pending_delete_id"]:
    st.sidebar.markdown("---")

if st.sidebar.button("➕ New Chat", key="new_chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.header("💬 Conversations")

# Display all conversation threads
if st.session_state["chat_threads"]:
    for thread_id in reversed(st.session_state["chat_threads"]):
        title = get_thread_title(thread_id) or "New Conversation"
        
        # Edit mode for this thread?
        if st.session_state["editing_thread_id"] == thread_id:
            edit_col1, edit_col2, edit_col3 = st.sidebar.columns([5, 1, 1])
            
            with edit_col1:
                new_title = st.text_input(
                    "Edit title",
                    value=title,
                    max_chars=50,
                    key=f"edit_input_{thread_id}",
                    label_visibility="collapsed",
                    placeholder="Enter title (max 50 chars)",
                )
            
            with edit_col2:
                if st.button("✓", key=f"save_{thread_id}", help="Save"):
                    if new_title and new_title.strip():
                        save_thread_title(thread_id, new_title.strip())
                        st.session_state["editing_thread_id"] = None
                        st.rerun()
            
            with edit_col3:
                if st.button("✗", key=f"cancel_edit_{thread_id}", help="Cancel"):
                    st.session_state["editing_thread_id"] = None
                    st.rerun()
        
        # Normal display mode
        else:
            col1, col2, col3 = st.sidebar.columns(
                [4, 1, 1], 
                vertical_alignment="center"
            )
            
            with col1:
                is_active = thread_id == st.session_state["thread_id"]
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(
                    title, 
                    key=f"thread_{thread_id}", 
                    use_container_width=True, 
                    type=btn_type
                ):
                    if thread_id != st.session_state["thread_id"]:
                        switch_thread(thread_id)
            
            with col2:
                if st.button("✏️", key=f"edit_{thread_id}", help="Edit title"):
                    st.session_state["editing_thread_id"] = thread_id
                    st.rerun()
            
            with col3:
                if st.button("❌", key=f"delete_{thread_id}", help="Delete conversation"):
                    st.session_state["pending_delete_id"] = thread_id
                    st.session_state["pending_delete_title"] = title
                    st.rerun()
else:
    st.sidebar.info("No conversations yet. Start chatting!")


# ============================================================================ #
# Main Chat Area
# ============================================================================ #

current_title = get_thread_title(st.session_state["thread_id"])
st.title(current_title)

# Show document indicator if file is uploaded
current_file = get_thread_filename(st.session_state["thread_id"])
if current_file:
    st.markdown(
        f"""
        <div class="file-indicator">
          <div class="fi-row">
            <div>📎</div>
            <div>
              <div><strong>Document loaded:</strong> {current_file}</div>
              <div><small>Ask me anything about this document!</small></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Stock analysis info banner
st.markdown(
    """
    <div class="stock-indicator">
      <div>
        <div><strong>📊 Stock Analysis Available</strong></div>
        <div><small>Ask me to analyze any NSE stock (e.g., "Analyze Titan stock" or "Give me HCL Technologies analysis")</small></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================================ #
# File Upload & Chat Input
# ============================================================================ #

popover_col, _ = st.columns([1, 5])

with popover_col:
    with st.popover("📎 Attach PDF", use_container_width=False):
        st.caption("Upload a PDF file (text-based PDFs only - scanned images won't work)")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key=f"file_uploader_{st.session_state['thread_id']}",
            label_visibility="collapsed",
            help="Attach a text-based PDF. Scanned documents without embedded text aren't supported yet.",
        )

user_input = st.chat_input("💬 Type your message here...")


# ============================================================================ #
# Process File Upload
# ============================================================================ #

if uploaded_file is not None:
    existing_file = get_thread_filename(st.session_state["thread_id"])
    already_uploaded = st.session_state["uploaded_once"].get(st.session_state["thread_id"])
    
    # Only process if it's a new file
    if (existing_file != uploaded_file.name) or (already_uploaded != uploaded_file.name):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            try:
                num_chunks = process_uploaded_file(
                    tmp_path,
                    file_type,
                    st.session_state["thread_id"],
                    uploaded_file.name,
                )
                
                if num_chunks <= 0:
                    st.error("No content extracted. The PDF might be empty or scanned without embedded text.")
                    st.stop()
                
                # Add success message to chat
                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": (
                            f"✅ Successfully processed **{uploaded_file.name}** "
                            f"({num_chunks} chunks created). Ask me anything about it!"
                        ),
                    }
                )
                
                st.success(f"Processed {num_chunks} chunks from {uploaded_file.name}")
                st.session_state["uploaded_once"][st.session_state["thread_id"]] = uploaded_file.name
                st.rerun()
            
            except FileExistsError as e:
                st.error(f"❌ {str(e)}")
                st.info("💡 **Tip:** Click '➕ New Chat' in the sidebar to create a fresh conversation for your new document.")
            
            except ValueError as e:
                st.error(f"❌ Error: {e}")
            
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.info("💡 If you're trying to upload a second document, please create a new chat instead.")
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ============================================================================ #
# Process User Input with Dynamic Progress
# ============================================================================ #

if user_input:
    current_thread = st.session_state["thread_id"]
    
    # Add user message to history
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        # Create containers for progress and response
        progress_container = st.empty()
        message_placeholder = st.empty()
        
        # Show progress updates
        progress_text = ""
        for progress_update in get_progress_updates(user_input, current_thread):
            if progress_update["step"] == "complete":
                # Clear progress when done
                progress_container.empty()
                break
            else:
                progress_text = progress_update["message"]
                progress_container.markdown(progress_text)
        
        # Stream the actual response
        full_response = ""
        token_stream, needs_rag = get_streaming_response(user_input, current_thread)
        
        for token in token_stream:
            full_response += token
            message_placeholder.markdown(full_response + "▌")  # Show cursor
        
        message_placeholder.markdown(full_response)
        
        # Save to history
        ai_message = full_response
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
        save_message_to_graph(current_thread, user_input, ai_message)
        
        # Auto-generate title for new conversations
        current_title = get_thread_title(current_thread)
        if current_title == "New Conversation":
            first_msgs = [
                HumanMessage(content=user_input), 
                AIMessage(content=ai_message)
            ]
            title = generate_title_from_conversation(first_msgs)
            save_thread_title(current_thread, title)
        
        st.rerun()


# ============================================================================ #
# Footer
# ============================================================================ #

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
      <p>Powered by LangGraph & Azure OpenAI</p>
      <p>📎 PDF Analysis | 📊 Stock Analysis (NSE)</p>
    </div>
    """,
    unsafe_allow_html=True,
)