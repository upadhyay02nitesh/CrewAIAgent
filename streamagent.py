import streamlit as st
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai_tools import SerpApiGoogleSearchTool
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

# App UI Setup
st.set_page_config(
    page_title="GenAI Latest Trends",
    page_icon="🚀",
    layout="wide"
)

# Header
st.title("GenAI Latest Trends & Updates")
st.caption("Discover cutting-edge AI developments from top companies")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "AI Model",
        ["mistralai/mistral-7b-instruct", "openai/gpt-3.5-turbo", "anthropic/claude-2"]
    )
    temperature = st.slider("Creativity Level", 0.0, 1.0, 0.5)
    companies = st.multiselect(
        "Companies to Track",
        ["OpenAI", "Google", "Microsoft", "AWS", "Meta", "Anthropic", "Mistral", "Cohere"],
        default=["OpenAI", "Google", "Microsoft"]
    )
    update_button = st.button("Get Latest Updates", type="primary")

# Main content area
if update_button:
    with st.spinner("🛰️ Scanning the AI universe for latest developments..."):
        # Initialize CrewAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=1000
        )

        web_search_tool = SerpApiGoogleSearchTool()
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Define Agents
        researcher = Agent(
            role="AI Trend Researcher",
            goal=f"Find latest AI announcements from {', '.join(companies)} in last 30 days",
            backstory="Expert researcher tracking cutting-edge AI updates",
            tools=[web_search_tool],
            verbose=False
        )

        summarizer = Agent(
            role="Tech Content Writer",
            goal="Create engaging summaries of AI developments",
            backstory="Skilled tech communicator who makes complex topics accessible",
            tools=[],
            verbose=False
        )

        # Define Tasks
        research_task = Task(
            description=f"Search for recent AI launches from {', '.join(companies)}",
            expected_output="Bullet list of company updates with tool names and brief descriptions",
            agent=researcher
        )

        summary_task = Task(
            description="Transform technical updates into engaging content",
            expected_output="Well-written summaries with company, tool name, and clear benefits",
            agent=summarizer,
            output_file='genai_trends.md'  # Added output file
        )

        # Assemble Crew
        crew = Crew(
            agents=[researcher, summarizer],
            tasks=[research_task, summary_task],
            verbose=False,
            planning=True
        )
        
        # Run the Crew and get the raw output
        results = crew.kickoff()
        text_output = str(results)  # Convert CrewOutput to string

    # Display Results
    st.success("✅ Latest AI Trends Discovered!")
    
    with st.expander("📊 Full Analysis", expanded=True):
        st.subheader(f"AI Trends as of {current_date}")
        st.markdown(text_output)  # Display as markdown
    
    # Generate shareable content
    st.subheader("📲 Shareable Content")
    st.code(text_output, language="markdown")
    
    # Fixed download button
    st.download_button(
        label="Download Summary",
        data=text_output,
        file_name=f"genai_trends_{current_date}.md",
        mime="text/markdown"
    )
else:
    st.info("👈 Configure your search in the sidebar and click 'Get Latest Updates'")

# Footer
st.markdown("---")
st.caption("Powered by CrewAI • Updated daily • Not affiliated with any companies mentioned")