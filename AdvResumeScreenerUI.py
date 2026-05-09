import asyncio
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pypdf import PdfReader

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ---------------------------
# MODEL CLIENT
# ---------------------------
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    model_info={
        "family": "gpt-4",
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
    }
)


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
# SAFE JSON EXTRACTION
# ---------------------------
def extract_json_from_text(content):
    if not content:
        return None

    try:
        return json.loads(content)
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except Exception:
            return None

    return None


def extract_agent_outputs(messages):
    parser_data = {}
    matcher_data = {}
    screening_data = {}
    summary = ""

    for msg in messages:
        content = getattr(msg, "content", "")

        if not content:
            continue

        data = extract_json_from_text(content)

        if isinstance(data, dict):
            if "years_of_experience" in data:
                parser_data = data

            elif "match_score" in data:
                matcher_data = data

            elif "decision" in data:
                screening_data = data

        else:
            if "TERMINATE" in content:
                summary = content.replace("TERMINATE", "").strip()
            else:
                summary = content.strip()

    return parser_data, matcher_data, screening_data, summary


# ---------------------------
# AGENT SETUP
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

Output MUST be valid JSON only:
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

    jd_matcher = AssistantAgent(
        name="JDMatcherAgent",
        model_client=model_client,
        system_message="""
You are a Job Description Matching Engine.

Responsibilities:
- Compare parsed resume data with job description.
- Identify match score and gaps.

Output MUST be valid JSON only:
{
  "match_score": number,
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

    screening_agent = AssistantAgent(
        name="ScreeningAgent",
        model_client=model_client,
        system_message="""
You are a Hiring Decision Engine.

Responsibilities:
- Decide whether the candidate should proceed.

Output MUST be valid JSON only:
{
  "decision": "HIRE" | "REJECT" | "HOLD",
  "confidence": number,
  "reasoning": ""
}

Rules:
- Base decision on match score and gaps.
- Be strict but fair.
- Avoid bias or assumptions.
"""
    )

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
- Keep it concise.
- Use professional HR language.
- No JSON output.

IMPORTANT:
After generating the final summary, respond with:
TERMINATE
Do not generate any further messages.
"""
    )

    return RoundRobinGroupChat(
        participants=[
            resume_parser,
            jd_matcher,
            screening_agent,
            summary_agent
        ],
        max_turns=4
    )


# ---------------------------
# RUN AGENTS
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

def build_skill_chart_data(matched_skills, missing_skills):
    return pd.DataFrame({
        "Category": ["Matched Skills", "Missing Skills"],
        "Count": [len(matched_skills), len(missing_skills)]
    })

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide"
)

st.title("📄 AI Resume Screening Dashboard")
st.caption("Upload a resume and compare it against a job description.")

jd = st.text_area(
    "Enter Job Description",
    height=220,
    placeholder="Example: Python developer with APIs, SQL, cloud, Docker..."
)

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if st.button("Analyze Candidate", type="primary"):

    if not jd or not uploaded_file:
        st.warning("Please provide both Job Description and Resume PDF.")

    else:
        with st.spinner("Analyzing candidate profile..."):

            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                st.error("Could not extract text from the PDF. Please upload a text-based resume PDF.")
                st.stop()

            groupchat = create_agents(model_client)
            messages = run_agents_sync(groupchat, jd, resume_text)

            parser_data, matcher_data, screening_data, summary = extract_agent_outputs(messages)

        st.success("Analysis Complete!")

        # ---------------------------
        # TOP METRICS
        # ---------------------------
        st.subheader("📊 Candidate Screening Snapshot")

        match_score = matcher_data.get("match_score", 0) or 0
        confidence = screening_data.get("confidence", 0) or 0
        decision = screening_data.get("decision", "N/A")
        experience = parser_data.get("years_of_experience", "N/A")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Match Score", f"{match_score}%")
        col2.metric("Decision", decision)
        col3.metric("Confidence", f"{confidence}%")
        col4.metric("Experience", f"{experience} yrs")

        st.divider()

        # ---------------------------
        # TABS
        # ---------------------------
        tab1, tab2, tab3 = st.tabs([
            "🧾 Summary",
            "🧠 Skills Analysis",
            "📈 Evaluation"
        ])

        # ---------------------------
        # SUMMARY
        # ---------------------------
        with tab1:
            st.subheader("Recruiter Summary")

            if summary:
                st.markdown(summary)
            else:
                st.info("No final summary was generated.")

            st.subheader("Screening Reasoning")
            st.write(screening_data.get("reasoning", "No reasoning available."))

        # ---------------------------
        # SKILLS ANALYSIS
        # ---------------------------
        with tab2:
            matched_skills = matcher_data.get("matched_skills", [])
            missing_skills = matcher_data.get("missing_skills", [])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Matched Skills")

                if matched_skills:
                    for skill in matched_skills:
                        st.success(skill)
                else:
                    st.info("No matched skills found.")

            with col2:
                st.subheader("❌ Missing Skills")

                if missing_skills:
                    for skill in missing_skills:
                        st.error(skill)
                else:
                    st.info("No missing skills found.")

            st.subheader("Skill Coverage")

            total_skills = len(matched_skills) + len(missing_skills)

            if total_skills > 0:
                coverage = int((len(matched_skills) / total_skills) * 100)
                st.progress(coverage / 100)
                st.write(f"Skill Coverage: **{coverage}%**")
            else:
                st.info("Skill coverage could not be calculated.")

        # ---------------------------
        # EVALUATION
        # ---------------------------
        with tab3:
            st.subheader("📈 Visual Evaluation")

            col1, col2 = st.columns(2)

            with col1:
                fig_match = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=match_score,
                    title={"text": "Resume Match Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "royalblue"},
                        "steps": [
                            {"range": [0, 40], "color": "#ffcccc"},
                            {"range": [40, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#d4edda"},
                        ],
                    }
                ))
                fig_match.update_layout(height=300)
                st.plotly_chart(fig_match, use_container_width=True)

            with col2:
                fig_confidence = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={"text": "Decision Confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "royalblue"},
                        "steps": [
                            {"range": [0, 40], "color": "#ffcccc"},
                            {"range": [40, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#d4edda"},
                        ],
                    }
                ))
                fig_confidence.update_layout(height=300)
                st.plotly_chart(fig_confidence, use_container_width=True)

            st.subheader("🧠 Skill Match Distribution")

            matched_skills = matcher_data.get("matched_skills", [])
            missing_skills = matcher_data.get("missing_skills", [])

            fig_skills = go.Figure(data=[
                go.Pie(
                    labels=["Matched Skills", "Missing Skills"],
                    values=[len(matched_skills), len(missing_skills)],
                    hole=0.55
                )
            ])

            fig_skills.update_layout(
                height=350,
                annotations=[
                    dict(
                        text="Skills",
                        x=0.5,
                        y=0.5,
                        font_size=20,
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig_skills, use_container_width=True)

            total_skills = len(matched_skills) + len(missing_skills)

            if total_skills > 0:
                coverage = int((len(matched_skills) / total_skills) * 100)
                st.progress(coverage / 100)
                st.write(f"Skill Coverage: **{coverage}%**")
            else:
                st.info("Skill coverage could not be calculated.")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("💪 Strengths")
                strengths = matcher_data.get("strengths", [])

                if strengths:
                    for strength in strengths:
                        st.write(f"✔️ {strength}")
                else:
                    st.info("No strengths found.")

            with col2:
                st.subheader("⚠️ Concerns")
                concerns = matcher_data.get("concerns", [])

                if concerns:
                    for concern in concerns:
                        st.write(f"⚠️ {concern}")
                else:
                    st.info("No concerns found.")           