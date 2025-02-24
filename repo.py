
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import github

# Install necessary packages
# !pip install langgraph langchain-core langchain-groq python-dotenv streamlit PyGithub

# Load environment variables from .env file
load_dotenv()

# Set Groq API key
os.environ["GROQ_API_KEY"] = ""  # Replace with your actual Groq API key

# Initialize the Groq model
model = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7)

# GitHub Authentication (Replace with your personal access token)
github_token = ""  # Store your token securely!
g = github.Github(github_token)


# Define the State class
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    github_repo: str  # Add field for GitHub repository


def make_code_review_graph():
    graph_workflow = StateGraph(State)

    def initial_analysis(state):
        # Placeholder for detailed analysis from GitHub, replace with the actual review method
        prompt = SystemMessage(content=f"Perform an initial code review of the following code, with focus on style, security and syntax:\n{state.get('github_repo', 'No repository provided')}")
        return {"messages": [model.invoke([prompt] + state["messages"])]}

    def human_review(state):
        prompt = SystemMessage(content=f"Considering the previous analysis, provide expert feedback considering the following code from: \n{state.get('github_repo', 'No repository provided')}. \nFocus on complex issues, edge cases and potential improvements.")
        return {"messages": [model.invoke([prompt] + state["messages"])]}

    def final_report(state):
        prompt = SystemMessage(content="Merge the initial analysis and human feedback into a comprehensive final code review report.")
        return {"messages": [model.invoke([prompt] + state["messages"])]}


    graph_workflow.add_node("initial_analysis", initial_analysis)
    graph_workflow.add_node("human_review", human_review)
    graph_workflow.add_node("final_report", final_report)

    graph_workflow.add_edge(START, "initial_analysis")
    graph_workflow.add_edge("initial_analysis", "human_review")
    graph_workflow.add_edge("human_review", "final_report")
    graph_workflow.add_edge("final_report", END)
    
    agent = graph_workflow.compile()
    return agent


agent = make_code_review_graph()

st.title("Code Review Agent")

repo_url = st.text_input("Enter GitHub Repository URL:")

if st.button("Review Code"):
    if repo_url:
        try:
            # Extract repository owner and name from URL
            repo_parts = repo_url.split('/')
            owner = repo_parts[-2]
            repo_name = repo_parts[-1]

            repo = g.get_repo(f"{owner}/{repo_name}")

            # Now, you can access various details about the repo
            #  For example, get the default branch
            default_branch = repo.default_branch
            # Get the content of a specific file
            # ... your code for fetching code files or relevant data

            initial_state = {"messages": [], "github_repo": repo_url}

            result = agent.invoke(initial_state)
            st.write(result["messages"][-1].content)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid GitHub repository URL.")
