
# Install necessary packages
#!pip install langgraph langchain-core langchain-groq python-dotenv streamlit

# Import necessary modules
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

# Set Groq API key
os.environ["GROQ_API_KEY"] = "g"  # Replace with your actual Groq API key

# Initialize the Groq model
model = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7)

# Define the State class
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the graph creation function
def make_blog_generation_graph():
    """Create a blog generation agent"""
    graph_workflow = StateGraph(State)

    # Define functions within the scope of make_blog_generation_graph
    def generate_title(state):
        prompt_1 = SystemMessage(content="As an experienced writer generate one blog title.")
        return {"messages": [model.invoke([prompt_1] + state["messages"])]}

    def generate_content(state):
        prompt_2 = SystemMessage(content="As an experienced content creator write a blog with 500 word limit in 4 paragraphs.")
        return {"messages": [model.invoke([prompt_2] + state["messages"])]}

    # Add nodes to the graph
    graph_workflow.add_node("title_generation", generate_title)
    graph_workflow.add_node("content_generation", generate_content)

    # Define graph edges
    graph_workflow.add_edge("title_generation", "content_generation")
    graph_workflow.add_edge("content_generation", END)
    graph_workflow.add_edge(START, "title_generation")

    # Compile the graph into an executable agent
    agent = graph_workflow.compile()

    return agent

# Create and Compile the Graph agent
agent = make_blog_generation_graph()

# Streamlit App
st.title("Blog Generator App")

# User Input
user_topic = st.text_input("Enter the blog topic:", "the history of chocolate")

# Generate Button
if st.button("Generate Blog"):
    if user_topic:
      with st.spinner("Generating blog..."):
          try:
            result = agent.invoke({"messages": [HumanMessage(content=f"Write a blog about {user_topic}")]})

            # Display the Results
            st.subheader("Generated Blog Title:")
            st.write(result['messages'][1].content) #Title

            st.subheader("Generated Blog Content:")
            st.write(result['messages'][2].content) #Content

          except Exception as e:
              st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a blog topic.")
