import os
import asyncio
import streamlit as st
from pypdf import PdfReader

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# api_key = st.secrets["OPENROUTER_API_KEY"]

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key = "sk-proj-***",
    model_info={
        "family": "gpt-4",
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
    }
)

# st.write("Key loaded:", bool(st.secrets.get("OPENROUTER_API_KEY")))

# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# ---------------------------
# AGENT SETUP (reuse yours)
# ---------------------------

def create_agents(model_client):

    resume_parser = AssistantAgent(
        name="ResumeParserAgent",
        model_client=model_client,
        system_message="""
    You are a Resume Parsing Specialist.

    Responsibilities:
    - Extract structured information from the resume.
    - Normalize data into consistent fields.

    Output MUST be in JSON format:
    {
    "years_of_experience": number,
    "skills": [],
    "technologies": [],
    "cloud_experience": [],
    "projects": [],
    "education": "",
    "summary": ""
    }

    Rules:
    - Do not infer beyond given data.
    - Keep output concise and factual.
    - If data is missing, return null.
    """
    )

    # ---------------------------
    # JD Matcher Agent
    # ---------------------------
    jd_matcher = AssistantAgent(
        name="JDMatcherAgent",
        model_client=model_client,
        system_message="""
    You are a Job Description Matching Engine.

    Responsibilities:
    - Compare parsed resume data with job description.
    - Identify match score and gaps.

    Output MUST be in JSON format:
    {
    "match_score": number (0-100),
    "matched_skills": [],
    "missing_skills": [],
    "strengths": [],
    "concerns": []
    }

    Rules:
    - Be objective and data-driven.
    - Do not hallucinate skills.
    - Base evaluation strictly on inputs.
    """
    )

    # ---------------------------
    # Screening Decision Agent
    # ---------------------------
    screening_agent = AssistantAgent(
        name="ScreeningAgent",
        model_client=model_client,
        system_message="""
    You are a Hiring Decision Engine.

    Responsibilities:
    - Decide whether the candidate should proceed.

    Output MUST be in JSON format:
    {
    "decision": "HIRE" | "REJECT" | "HOLD",
    "confidence": number (0-100),
    "reasoning": ""
    }

    Rules:
    - Base decision on match score and gaps.
    - Be strict but fair.
    - Avoid bias or assumptions.
    """
    )

    # ---------------------------
    # Final Summary Agent
    # ---------------------------
    summary_agent = AssistantAgent(
        name="SummaryAgent",
        model_client=model_client,
        system_message="""
    You are a Recruiter-Facing Summary Generator.

    Responsibilities:
    - Convert all prior outputs into a professional summary.

    Output Format:
    - Candidate Overview
    - Key Strengths
    - Skill Gaps
    - Final Recommendation

    Rules:
    - Keep it concise (max 150 words).
    - Use professional HR language.
    - No JSON output.

    IMPORTANT:
    After generating the final summary, respond with:
    TERMINATE
    Do not generate any further messages.
    """
    )

    # ---------------------------
    # Group Chat
    # ---------------------------
    return RoundRobinGroupChat(
        participants=[
            resume_parser,
            jd_matcher,
            screening_agent,
            summary_agent
        ],
        max_turns=4  # prevents infinite loops
    )


# ---------------------------
# RUN AGENTS (ASYNC WRAPPER)
# ---------------------------
async def run_agents(groupchat, jd, resume_text):
    result = []

    async for message in groupchat.run_stream(
        task=f"""
Job Description:
{jd}

Resume:
{resume_text}
"""
    ):
        result.append(message)

    return result


def run_agents_sync(groupchat, jd, resume_text):
    return asyncio.run(run_agents(groupchat, jd, resume_text))


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("📄 Resume Screening System")

# Job Description input
jd = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="Python developer with APIs, SQL, cloud..."
)

# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if st.button("Analyze Candidate"):

    if not jd or not uploaded_file:
        st.warning("Please provide both JD and Resume.")
    else:
        with st.spinner("Processing..."):

            # Extract text
            resume_text = extract_text_from_pdf(uploaded_file)

            # Create agents
            groupchat = create_agents(model_client)

            # Run agents
            messages = run_agents_sync(groupchat, jd, resume_text)

        st.success("Analysis Complete!")

        # ---------------------------
        # DISPLAY RESULTS
        # ---------------------------
        st.subheader("📊 Agent Outputs")

        final_msg = ""

        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if content:
                final_msg = content
                break       

        st.subheader("🧾 Final Recommendation")
        st.markdown(final_msg)