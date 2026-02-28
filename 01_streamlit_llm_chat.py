import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(page_title="Local LLM Chat", layout="wide")

# Title
st.title("🤖 Local LLM Chat Interface")
st.markdown("---")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    model = st.selectbox(
        "Select Model",
        ["llama3", "llama2", "mistral"],
        help="Choose the Ollama model to use"
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Lower values make output more focused, higher values more creative"
    )
    st.markdown("---")
    st.info("Make sure Ollama is running locally with the selected model installed.")

# System context
system_context = """You are an experienced education instructor specializing in technology and computer science. 
Your role is to explain technical concepts clearly and concisely, breaking down complex ideas into understandable parts. 
Provide examples when applicable and ensure your explanations are suitable for students at various skill levels."""

# Initialize the LLM
@st.cache_resource
def get_llm(model_name):
    return OllamaLLM(model=model_name)

llm = get_llm(model)

# Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_context),
    ("human", "{question}")
])

chain = prompt_template | llm

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about technology and learning... (type 'quit', 'exit', or 'end' to clear chat)"):
    # Check for quit commands
    quit_commands = ["quit", "exit", "end", "bye", "goodbye"]
    
    if prompt.lower().strip() in quit_commands:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display goodbye message
        with st.chat_message("assistant"):
            goodbye_message = "Thank you for chatting with me! Your conversation has been cleared. Feel free to start a new conversation anytime. 👋"
            st.markdown(goodbye_message)
            st.session_state.messages.append({"role": "assistant", "content": goodbye_message})
        
        # Clear chat history after a brief moment
        st.session_state.messages = []
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is running and the model is installed: `ollama pull " + model + "`")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by LangChain + Ollama</p>",
    unsafe_allow_html=True
)
