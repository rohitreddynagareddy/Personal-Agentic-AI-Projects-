# Install necessary packages
!pip install langgraph langchain-core langchain-groq python-dotenv streamlit diffusers transformers accelerate torch

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from diffusers import StableDiffusionPipeline
import torch
import io
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Set Groq API key
# Replace 'YOUR_GROQ_API_KEY' with your actual Groq API key.
os.environ["GROQ_API_KEY"] = "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"  # Add your key here

# Initialize the Groq model
model = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7)

# Determine if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Stable Diffusion model
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe = pipe.to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32)
    print(
        "No GPU detected. Image generation will be very slow. Consider using a GPU-enabled environment for faster results."
    )

# New functions for the automated workflow
def generate_storyboard(description):
    prompt = f"Generate a detailed storyboard for a video based on this description:\n{description}"
    response = model.invoke([SystemMessage(content=prompt)])
    return response.content

def generate_image_prompt(scene_description):
    prompt = f"Create a detailed prompt for an image generation AI model based on this scene description: {scene_description}"
    response = model.invoke([SystemMessage(content=prompt)])
    return response.content

# Define the State class
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


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

# Function to run the generation process and display results (within Colab)
def run_video_generator(user_topic):
    print(f"Generating video content for: {user_topic}")

    if user_topic:  
        print("Generating storyboard...")
        storyboard = generate_storyboard(user_topic)
        print("StoryBoard:\n", storyboard)

        scenes = storyboard.split("SCENE") 

        for scene in scenes:
            if scene.strip(): 
                image_prompt = generate_image_prompt(scene)
                print(f"Image Prompt for this scene: {image_prompt}")

                print("Generating Image ...")
                try:
                    image = pipe(image_prompt).images[0]

                    # Display the image in Colab
                    display(image)
                except Exception as e:
                    print(f"An error occurred while generating image: {e}")
        # Create blog and save it
        result = agent.invoke({"messages": [HumanMessage(content=f"Write a blog about {user_topic}")]})
        blog_title = result['messages'][1].content
        blog_content = result['messages'][2].content
        print("Blog generated")
        print(f"Blog Title:{blog_title}")
        print(f"Blog content:{blog_content}")

        # Create Markdown content
        markdown_content = f"# {blog_title}\n\n{blog_content}"

        # Save to file in Colab
        file_name = f"{user_topic.replace(' ', '_')}.md"
        with open(file_name, "w") as f:
            f.write(markdown_content)
        print(f"Blog generated and saved to {file_name}")

    else:
        print("Please enter a topic first.")

# Example usage:
user_topic = "the history of chocolate"  # Replace with your desired topic
run_video_generator(user_topic)