from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerpApiGoogleSearchTool
from datetime import datetime
import os

# Load environment variables from .env
load_dotenv()

# Setup API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Define LLM
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.5,
    max_tokens=1000
)

# Define Web Search Tool
web_search_tool = SerpApiGoogleSearchTool()

# Define Agents
researcher = Agent(
    role="AI Trend Researcher",
    goal=(
        f"Explore the latest Generative AI and Agentic AI developments from the last 30 days. "
        f"Focus on announcements, tools, APIs, or product launches from major AI companies like "
        f"OpenAI, Google, Microsoft, AWS, Meta, Anthropic, Mistral, and Cohere. "
        f"Also include trending new-gen AI tools or ecosystems like MCP Server, Cursor, AutoGen Studio, LangGraph, "
        f"OpenDevin, Devika, Code Interpreter, and other agentic systems. "
        f"Return only real products or announcements with a clear one-line summary. Today is {current_date}."
    ),
    backstory=(
        "You're an expert AI researcher who tracks the most recent innovations using live web data. "
        "Your mission is to uncover real, valuable updates that help AI enthusiasts and developers stay current. "
        "Ignore rumors or speculation — only provide verifiable updates and tools."
    ),
    tools=[web_search_tool],
    verbose=False
)

summarizer = Agent(
    role="One-line Writer",
    goal="Summarize each latest AI tool announcement in one crisp, readable line mentioning company and tool.",
    backstory="A crisp and creative tech summarizer who makes even complex tool announcements sound simple in one line.",
    tools=[],
    verbose=False
)

# Define Tasks
research_task = Task(
    description=(
        "Use the web to search for the most recent tool or technology launches in Generative AI and Agentic AI "
        "from OpenAI, Google, Microsoft, AWS, Meta, Anthropic, Mistral, and Cohere. "
        "Summarize each company's key update (tool name, release, or announcement)."
    ),
    expected_output=(
        "A list of 5 to 8 bullet points, each with: [Company] - [Tool/Tech Name] - [Short Summary or Use]."
    ),
    agent=researcher
)

summary_task = Task(
    description=(
        "Read the researcher's bullet list and rewrite it as a beautifully written content summary. "
        "For each tool or development, write at least **three engaging lines** in a simple, human tone. "
        "Make it feel like a fresh LinkedIn post or tech blog paragraph – highlight the **company**, the **tool name**, and explain **what it does and why it matters**. "
        "Avoid technical jargon. Keep it friendly, informative, and interesting for general readers. "
        "Use short paragraphs, relatable language, and make each tool feel exciting or useful."
    ),
    expected_output="Short, simple, engaging summaries for each AI innovation, at least 3 lines each.",
    agent=summarizer,
    output_file='blog-posts/genai_agentic_ai_july.md'
)


# Assemble Crew
crew = Crew(
    agents=[researcher, summarizer],
    tasks=[research_task, summary_task],
    verbose=False,
    planning=True
)

# Run the Crew
crew.kickoff()
