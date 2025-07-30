import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# ---------- SETUP ----------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="📄 Resume Analyzer AI", layout="wide")
st.title("📄 Resume Analyzer & Summary Generator with CrewAI")

# ---------- HELPER ----------
def load_resume_from_uploaded_file(uploaded_file):
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp_resume.pdf")
    docs = loader.load()
    return " ".join([doc.page_content for doc in docs])

def generate_outputs(resume_text):
    # LLM
    llm = ChatOpenAI(
        model="openrouter/mistralai/mistral-7b-instruct",
        temperature=0.3,
        max_tokens=1000
    )

    # AGENTS
    summary_agent = Agent(
        role="Professional Summary Writer",
        goal="Write a concise 3-4 line summary for the top of the resume",
        backstory="Expert in summarizing resumes to highlight key strengths, skills, and goals.",
        llm=llm,
        verbose=False
    )

    linkedin_agent = Agent(
        role="LinkedIn Profile Expert",
        goal="Create an engaging LinkedIn summary that showcases professional achievements",
        backstory="Specialist in crafting compelling LinkedIn profiles that attract recruiters",
        llm=llm,
        verbose=False
    )

    naukri_agent = Agent(
        role="Naukri Profile Specialist",
        goal="Write an effective summary for Indian job portals like Naukri.com",
        backstory="Expert in creating job portal profiles that get noticed by Indian recruiters",
        llm=llm,
        verbose=False
    )

    improver_agent = Agent(
        role="Resume Improvement Expert",
        goal="Identify areas for improvement in the resume",
        backstory="Professional resume reviewer with experience in optimizing resumes for ATS",
        llm=llm,
        verbose=False
    )

    # TASKS
    summary_task = Task(
        description=f"Write a 3-4 line professional summary for a resume based on:\n\n{resume_text}",
        expected_output="Concise professional summary (3-4 lines)",
        agent=summary_agent,
        output_file="summary_output.txt"
    )

    linkedin_task = Task(
        description=f"Create a LinkedIn 'About' section summary based on:\n\n{resume_text}",
        expected_output="Engaging LinkedIn summary (5-6 lines) with a professional tone",
        agent=linkedin_agent,
        output_file="linkedin_output.txt"
    )

    naukri_task = Task(
        description=f"Write a Naukri.com profile summary based on:\n\n{resume_text}",
        expected_output="Detailed Naukri profile summary (150-200 words) with relevant keywords",
        agent=naukri_agent,
        output_file="naukri_output.txt"
    )

    improver_task = Task(
        description=f"Analyze this resume and suggest improvements:\n\n{resume_text}",
        expected_output="Bullet-point list of actionable resume improvement suggestions",
        agent=improver_agent,
        output_file="improver_output.txt"
    )

    # CREW
    crew = Crew(
        agents=[summary_agent, linkedin_agent, naukri_agent, improver_agent],
        tasks=[summary_task, linkedin_task, naukri_task, improver_task],
        verbose=True
    )

    # Execute the crew
    crew.kickoff()
    
    # Collect all outputs
    outputs = []
    for output_file in ["summary_output.txt", "linkedin_output.txt", "naukri_output.txt", "improver_output.txt"]:
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                outputs.append(f.read())
        else:
            outputs.append("Output not available")
    
    return outputs

# ---------- STREAMLIT ----------
uploaded_file = st.file_uploader("📤 Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file:
    st.success("✅ Resume uploaded successfully!")
    st.subheader("📘 File Preview")
    
    with st.expander("View Resume Text"):
        resume_text = load_resume_from_uploaded_file(uploaded_file)
        if resume_text:
            st.write(resume_text)
        else:
            st.error("Failed to extract text from PDF")
            st.stop()

    if st.button("🚀 Generate Resume Analysis"):
        with st.spinner("🤖 AI Agents are analyzing your resume..."):
            try:
                results = generate_outputs(resume_text)
            except Exception as e:
                st.error(f"Error generating analysis: {e}")
                st.stop()

        if results and len(results) == 4:
            st.success("🎉 Analysis completed successfully!")
            
            # Professional Summary
            st.subheader("📝 Professional Summary")
            st.markdown(f"```\n{results[0]}\n```")
            st.download_button(
                label="📥 Download Professional Summary",
                data=results[0],
                file_name="professional_summary.txt",
                mime="text/plain"
            )
            
            # LinkedIn Summary
            st.subheader("🔗 LinkedIn Summary")
            st.markdown(f"```\n{results[1]}\n```")
            st.download_button(
                label="📥 Download LinkedIn Summary",
                data=results[1],
                file_name="linkedin_summary.txt",
                mime="text/plain"
            )
            
            # Naukri Summary
            st.subheader("💼 Naukri Profile Summary")
            st.markdown(f"```\n{results[2]}\n```")
            st.download_button(
                label="📥 Download Naukri Summary",
                data=results[2],
                file_name="naukri_summary.txt",
                mime="text/plain"
            )
            
            # Improvement Suggestions
            st.subheader("🛠️ Resume Improvement Suggestions")
            st.markdown(f"```\n{results[3]}\n```")
            st.download_button(
                label="📥 Download Improvement Suggestions",
                data=results[3],
                file_name="resume_improvements.txt",
                mime="text/plain"
            )
        else:
            st.error("Failed to generate complete analysis")