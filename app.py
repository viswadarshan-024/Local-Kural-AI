import streamlit as st
import pandas as pd
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
from typing import List, Dict, Any, Optional
import uuid

# Set page configuration
st.set_page_config(
    page_title="ThirukkuralAI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
# TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "your-tavily-api-key")
TAVILY_API_KEY = "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"
SYSTEM_PROMPT = """You are ThirukkuralAI, an AI assistant specialized in Thirukkural and Tamil culture. 
You provide insights from the ancient Tamil text Thirukkural written by Thiruvalluvar.
Always be respectful, helpful, and provide accurate information related to Thirukkural and Tamil culture.
Format your responses clearly with proper sections when presenting Thirukkural couplets.
"""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #FF5733;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 20px;
        color: #808080;
        text-align: center;
        margin-bottom: 30px;
    }
    .kural-box {
        background-color: #f0f8ff;
        border-left: 5px solid #FF5733;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .kural-tamil {
        font-weight: bold;
        font-size: 18px;
        color: #800000;
    }
    .kural-english {
        font-style: italic;
        color: #2F4F4F;
    }
    .kural-meaning {
        margin-top: 5px;
        color: #4B0082;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
    }
    .user-message {
        background-color: #E6F7FF;
        border-left: 5px solid #1E90FF;
    }
    .assistant-message {
        background-color: #F0FFF0;
        border-left: 5px solid #32CD32;
    }
    .message-content {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_models():
    import requests
    import json
    
    # Load the sentence transformer model for embedding generation
    # Use a smaller local model to avoid download issues
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    
    # We'll create placeholder objects for the tokenizer and model
    # since we'll use Ollama's API directly instead
    class OllamaTokenizer:
        def __init__(self):
            pass
        
        def __call__(self, text, return_tensors=None):
            return {"input_text": text}
            
        def decode(self, outputs, skip_special_tokens=True):
            return outputs if isinstance(outputs, str) else ""
    
    class OllamaModel:
        def __init__(self):
            self.device = "cpu"
            self.ollama_url = "http://localhost:11434/api/generate"
            
        def generate(self, input_text=None, max_length=512, num_return_sequences=1, 
                    temperature=0.7, top_p=0.9, do_sample=True):
            if not input_text:
                return ["No input provided"]
                
            try:
                payload = {
                    "model": "gemma3:1b",
                    "prompt": input_text["input_text"],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_length
                    }
                }
                
                response = requests.post(self.ollama_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    return result["response"]
                else:
                    return f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                return f"Error generating response: {str(e)}"
    
    tokenizer = OllamaTokenizer()
    model = OllamaModel()
    
    return sbert_model, model, tokenizer

@st.cache_resource
def load_vector_database():
    # Load the Thirukkural data
    df = pd.read_pickle("thirukkural_data.pkl")
    
    # Load the FAISS indices
    tamil_index = faiss.read_index("thirukkural_tamil_index.faiss")
    english_index = faiss.read_index("thirukkural_english_index.faiss")
    
    return df, tamil_index, english_index

def query_tavily(query: str) -> Dict[str, Any]:
    """Query Tavily Search API for web search results."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_domains": [],
        "exclude_domains": []
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying Tavily: {str(e)}")
        return {"answer": "", "results": []}

def semantic_search(query: str, sbert_model, tamil_index, english_index, df, language="english", top_k=5):
    """Perform semantic search on the vector database."""
    # Generate embedding for the query
    query_embedding = sbert_model.encode([query])[0]
    
    # Normalize the embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Choose the appropriate index based on language
    index = tamil_index if language == "tamil" else english_index
    
    # Search the index
    distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
    
    # Get the results
    results = []
    for i, idx in enumerate(indices[0]):
        kural_info = {
            "index": int(idx),
            "similarity": float(distances[0][i]),
            "kural_number": df.iloc[idx]["Number"],
            "kural_tamil": df.iloc[idx]["Kural"],
            "kural_english": df.iloc[idx]["Couplet"],
            "kural_tamil_meaning": df.iloc[idx]["Vilakam"],
            "kural_english_meaning": df.iloc[idx]["M_Varadharajanar"],
            "chapter": df.iloc[idx]["Chapter"],
            "section": df.iloc[idx]["Section"]
        }
        results.append(kural_info)
    
    return results

def generate_response(prompt: str, model, tokenizer, max_length=512):
    """Generate a response using the Ollama model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response
    response = model.generate(inputs, max_length=max_length, 
                             num_return_sequences=1, temperature=0.7, 
                             top_p=0.9, do_sample=True)
    
    # Remove the input prompt from the response if possible
    if isinstance(response, str) and response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def format_kural_results(results: List[Dict[str, Any]]) -> str:
    """Format the Kural search results for display."""
    formatted_text = ""
    
    for result in results:
        formatted_text += f"**Kural #{result['kural_number']}** (Chapter: {result['chapter']}, Section: {result['section']})\n\n"
        formatted_text += f"**Tamil:**\n{result['kural_tamil']}\n\n"
        formatted_text += f"**English Translation:**\n{result['kural_english']}\n\n"
        formatted_text += f"**Meaning:**\n{result['kural_english_meaning']}\n\n"
        formatted_text += "---\n\n"
    
    return formatted_text

def process_user_query(user_query: str, sbert_model, tamil_index, english_index, df, model, tokenizer) -> str:
    """Process the user query and generate an appropriate response."""
    
    # Determine the query type
    if any(greeting in user_query.lower() for greeting in ["hello", "hi", "hey", "greetings", "vanakkam"]):
        return "Vanakkam! I'm ThirukkuralAI, your guide to the ancient wisdom of Thirukkural. How can I help you explore the teachings of Thiruvalluvar today?"
    
    if any(about in user_query.lower() for about in ["who are you", "about you", "what can you do", "how do you work"]):
        return "I am ThirukkuralAI, an AI assistant specializing in Thirukkural and Tamil culture. I can help you find relevant Thirukkural couplets for life situations, explain Tamil cultural concepts, and provide insights from this ancient wisdom. I use semantic search to find the most relevant Thirukkural for your questions and can also search the web for information about Tamil culture and history."
    
    # Check if the query is related to Thirukkural or Tamil culture
    tamil_related = any(term in user_query.lower() for term in 
                       ["thirukkural", "thirukural", "kural", "thiruvalluvar", "tamil", "tamizh"])
    
    if tamil_related or "culture" in user_query.lower():
        # First try to get relevant kurals
        english_results = semantic_search(user_query, sbert_model, tamil_index, english_index, df, "english", top_k=3)
        
        # If no good results from semantic search, try web search
        if not english_results or english_results[0]["similarity"] < 0.6:
            # Prepare context from web search
            tavily_results = query_tavily(f"Thirukkural or Tamil culture information about: {user_query}")
            web_context = tavily_results.get("answer", "") + "\n\n"
            
            for result in tavily_results.get("results", [])[:3]:
                web_context += f"{result.get('title', '')}: {result.get('content', '')}\n\n"
            
            # Generate response using the model with web search context
            prompt = f"""
            {SYSTEM_PROMPT}
            
            Web search information:
            {web_context}
            
            User query: {user_query}
            
            Please provide a helpful response about Thirukkural or Tamil culture related to the query.
            """
            
            return generate_response(prompt, model, tokenizer)
        else:
            # Format the found kurals
            kural_context = format_kural_results(english_results)
            
            # Generate response using the model with kural context
            prompt = f"""
            {SYSTEM_PROMPT}
            
            User query: {user_query}
            
            Here are the relevant Thirukkural couplets I found:
            
            {kural_context}
            
            Please provide a thoughtful response that incorporates these Thirukkurals and addresses the user's query.
            """
            
            return generate_response(prompt, model, tokenizer)
    else:
        # For general life situations, find relevant Thirukkural
        english_results = semantic_search(user_query, sbert_model, tamil_index, english_index, df, "english", top_k=3)
        
        # Format the found kurals
        kural_context = format_kural_results(english_results)
        
        # Generate response using the model with kural context
        prompt = f"""
        {SYSTEM_PROMPT}
        
        User query about a life situation: {user_query}
        
        Here are the relevant Thirukkural couplets I found:
        
        {kural_context}
        
        Please provide wisdom and advice based on these Thirukkurals that addresses the user's situation.
        """
        
        return generate_response(prompt, model, tokenizer)

def main():
    # Display header
    st.markdown('<div class="main-header">ThirukkuralAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Wisdom of Thiruvalluvar for Modern Life</div>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        sbert_model, gemma_model, tokenizer = load_models()
        df, tamil_index, english_index = load_vector_database()
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><div class="message-content">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><div class="message-content">{message["content"]}</div></div>', unsafe_allow_html=True)
    
    # User input
    user_query = st.text_input("Ask about Thirukkural or Tamil culture, or share a life situation:", key="user_input")
    
    # Process query on submit
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        st.markdown(f'<div class="chat-message user-message"><div class="message-content">{user_query}</div></div>', unsafe_allow_html=True)
        
        # Process the query and get response
        with st.spinner("Consulting the wisdom of Thiruvalluvar..."):
            response = process_user_query(user_query, sbert_model, tamil_index, english_index, df, gemma_model, tokenizer)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display assistant response
        st.markdown(f'<div class="chat-message assistant-message"><div class="message-content">{response}</div></div>', unsafe_allow_html=True)
        
        # Clear the input box after processing
        st.rerun()
    
    # Sidebar with information
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Thiruvalluvar_Statue_at_Kanyakumari.jpg/330px-Thiruvalluvar_Statue_at_Kanyakumari.jpg", width=200)
        st.markdown("### About Thirukkural")
        st.markdown("""
        Thirukkural is a classic Tamil text consisting of 1,330 couplets (kurals) organizing the wisdom of life into three sections:
        
        1. **Virtue** (Aram) - 380 kurals
        2. **Wealth** (Porul) - 700 kurals
        3. **Love** (Inbam) - 250 kurals
        
        Written by Thiruvalluvar around the 5th century BCE, it provides timeless guidance on ethics, politics, economics, and love.
        """)
        
        st.markdown("### How to use this app")
        st.markdown("""
        - Ask specific questions about Thirukkural
        - Share a life situation for relevant kural guidance
        - Inquire about Tamil culture and philosophy
        - The app uses AI to find the most relevant wisdom from Thirukkural
        """)
        
        # Add clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()