import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew
from crewai.enums import ProcessType
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Load resume text
def load_resume(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return " ".join([doc.page_content for doc in docs])

RESUME_PATH = "CrewAI/resume/Nitesh.pdf"
resume_text = load_resume(RESUME_PATH)

# Define LLM
llm = ChatOpenAI(
    model="openrouter/mistralai/mistral-7b-instruct",
    temperature=0.3,
    max_tokens=1000
)

# ---------- AGENTS ----------
summary_agent = Agent(
    role="Professional Summary Writer",
    goal="Write a concise 3-4 line summary for the top of the resume",
    backstory="Expert in summarizing resumes to highlight key strengths, skills, and goals.",
    llm=llm,
    verbose=False
)

linkedin_agent = Agent(
    role="LinkedIn Summary Expert",
    goal="Generate an engaging LinkedIn summary for the user",
    backstory="You specialize in crafting professional and network-oriented LinkedIn summaries.",
    llm=llm,
    verbose=False
)

naukri_agent = Agent(
    role="Naukri Profile Summary Expert",
    goal="Write a job portal-friendly summary for Naukri profile",
    backstory="You help users attract recruiters on Naukri by using keywords and concise achievements.",
    llm=llm,
    verbose=False
)

suggestion_agent = Agent(
    role="Resume Improvement Advisor",
    goal="Suggest improvements and missing elements in the resume",
    backstory="You're a resume expert helping candidates enhance their resume with modern practices.",
    llm=llm,
    verbose=False
)

# ---------- TASKS ----------
summary_task = Task(
    description=(
        "Based on the following resume, write a 3–4 line impactful professional summary. "
        "Mention years of experience, core strengths, industries worked in, and future goals if mentioned.\n\n"
        f"Resume:\n{resume_text}"
    ),
    expected_output="A short, powerful professional summary (3–4 lines).",
    agent=summary_agent,
    output_file="analyzer_outputs/professional_summary.md"
)

linkedin_task = Task(
    description=(
        "Create a LinkedIn-friendly summary based on this resume. "
        "Make it slightly more conversational, highlight achievements, personality, and professional goals.\n\n"
        f"Resume:\n{resume_text}"
    ),
    expected_output="LinkedIn About section content (max 5–6 lines).",
    agent=linkedin_agent,
    output_file="analyzer_outputs/linkedin_summary.md"
)

naukri_task = Task(
    description=(
        "Craft a summary suitable for a Naukri.com profile based on this resume. "
        "Focus on roles, years of experience, key tech skills, and use industry-specific keywords.\n\n"
        f"Resume:\n{resume_text}"
    ),
    expected_output="Naukri Profile Summary (150–200 words).",
    agent=naukri_agent,
    output_file="analyzer_outputs/naukri_summary.md"
)

suggestion_task = Task(
    description=(
        "Analyze the resume content and suggest improvements. Include missing sections, grammar, formatting, or impact enhancements.\n\n"
        f"Resume:\n{resume_text}"
    ),
    expected_output="A list of suggestions to improve the resume.",
    agent=suggestion_agent,
    output_file="analyzer_outputs/resume_suggestions.md"
)

# ---------- CREW ----------
crew = Crew(
    agents=[summary_agent, linkedin_agent, naukri_agent, suggestion_agent],
    tasks=[summary_task, linkedin_task, naukri_task, suggestion_task],
    verbose=True,
    process_type=ProcessType.parallel,  # Run tasks in parallel
)

# ---------- EXECUTE ----------
crew.kickoff()
