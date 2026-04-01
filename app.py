from flask import Flask, render_template, request
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel 
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


summarize_prompt = PromptTemplate.from_template(
    "Summarize these tasks in one single sentence. "
    "Do not include any introductory phrases: {tasks}"
)

classify_prompt = PromptTemplate.from_template(
    "Categorize the following tasks into Work and Personal groups. "
    "Instructions: Do not use any bolding or asterisks (**). "
    "Do not say 'Sure' or 'Here are your tasks'. "
    "Format as 'Category Name:' followed by the list.\nTasks:\n{tasks}"
)

priority_prompt = PromptTemplate.from_template(
    "Assign High, Medium, or Low priority to these tasks. "
    "Instructions: Do not use bolding or asterisks (**). "
    "Provide only the list of tasks with their priority level. "
    "No conversational filler or introductory text.\nTasks:\n{tasks}"
)

chain = RunnableParallel({
    "summary": summarize_prompt | llm | StrOutputParser(),
    "categories": classify_prompt | llm | StrOutputParser(),
    "priority": priority_prompt | llm | StrOutputParser()
})

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    original_tasks = ""
    if request.method == "POST":
        original_tasks = request.form.get("tasks")
        if original_tasks:
           
            result = chain.invoke({"tasks": original_tasks})
    
    return render_template("index.html", result=result, original_tasks=original_tasks)

if __name__ == "__main__":
    app.run(debug=True)