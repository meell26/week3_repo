from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def count_words(text: str) -> str:
    return f"Number of words: {len(text.split())}"

def summarize(text: str) -> str:
    return "Summary: The day consisted of a workout, 8 hours of work, and cooking a healthy meal."

tools = [
    Tool(name="word_counter", func=count_words, description="Counts words in a string"),
    Tool(name="summarizer", func=summarize, description="Summarizes daily tasks")
]

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(llm, tools=tools)

user_input = input("Enter your task:\n")


result = agent.invoke({
    "messages": [{"role": "user", "content": user_input}]
})

print("\nAgent Result:")
print(result["messages"][-1].content)