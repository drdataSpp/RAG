from langchain_ollama import ChatOllama
from langchain.agents import create_agent


model_mistral = ChatOllama(
    model="mistral:7b",
    base_url="http://localhost:11434"
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model_mistral,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response['messages'][0]['content'])