import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Function to set up the vector store
def setup_vectorstore():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(working_dir, "vector_db_dir")  # Use absolute path to vector_db_dir
    
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings()
    
    # Check if the vector store directory exists
    if os.path.exists(persist_directory):
        # Load existing vector store with embedding function
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Loaded existing vector store.")
    else:
        st.error("Vector store not found. Please run vectorize_doc.py first.")
        st.stop()
    
    return vectorstore

# Function to create a conversational chain
def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    return chain

# Streamlit configuration
st.set_page_config(page_title="Multi Doc Chat", page_icon="📚", layout="centered")

# Prompt for API key at runtime
GROQ_API_KEY = st.text_input("Please enter your GROQ API Key:", type="password")

# Stop the app if no API key is provided
if not GROQ_API_KEY:
    st.warning("API Key is required to proceed.")
    st.stop()

# Set the API key as an environment variable for further use in the code
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

st.title("📚 Multimodal RAG")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load vector store only once
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

# Load conversational chain only once
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for queries
user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Use the conversational chain to process the user input
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
