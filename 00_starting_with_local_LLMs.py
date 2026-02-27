from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Ollama LLM with llama3 model
llm = OllamaLLM(model="llama3")

# Define the system context for an education instructor
system_context = """You are an experienced education instructor specializing in technology and computer science. 
Your role is to explain technical concepts clearly and concisely, breaking down complex ideas into understandable parts. 
Provide examples when applicable and ensure your explanations are suitable for students at various skill levels."""

# Define the question
question = "What is LLM?"

# Create a prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_context),
    ("human", "{question}")
])

# Create the chain
chain = prompt_template | llm

# Get the response
print("=" * 80)
print(f"Question: {question}")
print("=" * 80)
print("\nResponse from llama3:\n")

response = chain.invoke({"question": question})

print(response)
print("\n" + "=" * 80)
