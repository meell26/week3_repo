from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel 
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

summarize_prompt = PromptTemplate.from_template(
    "Summarize the general nature of these tasks in one sentence: {tasks}"
)

classify_prompt = PromptTemplate.from_template(
    "Separate these tasks into 'Work' and 'Personal' categories:\n{tasks}"
)

priority_prompt = PromptTemplate.from_template(
    "Assign a priority (High, Medium, Low) to each of these tasks: {tasks}"
)

chain = RunnableParallel({
    "summary": summarize_prompt | llm | StrOutputParser(),
    "categories": classify_prompt | llm | StrOutputParser(),
    "priority": priority_prompt | llm | StrOutputParser()
})

user_tasks = """
Finish report
Prepare slides
Buy groceries
Schedule meeting
Reply to emails
"""


result = chain.invoke({"tasks": user_tasks})

print("\n--- AI Task Planner Output ---")
print(f"Summary:\n{result['summary']}\n")
print(f"Categories:\n{result['categories']}\n")
print(f"Priority:\n{result['priority']}")