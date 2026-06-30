"""
Resume Optimizer AI -- Streamlit web app
------------------------------------------
Same prompt engineering and JSON contract as the CLI version, wrapped in a
browser UI. Deploy free on Streamlit Community Cloud.

Local run:
    pip install -r requirements.txt
    export GEMINI_API_KEY=...
    streamlit run app.py

Deployed run (Streamlit Cloud):
    Set GEMINI_API_KEY in the app's "Secrets" panel instead of an env var.
"""

import json
import os

import streamlit as st
from google import genai

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an ATS (Applicant Tracking System) simulator combined with a senior \
FAANG technical recruiter with 15 years of hiring experience. You evaluate resumes the way \
both a parsing algorithm and a human reviewer would.

Your tasks:
1. Score ATS compatibility (0-100) based on formatting, keyword density, section structure,
   and parseability.
2. If a job description is provided, score job match (0-100) based on skill overlap,
   seniority alignment, and keyword coverage. If no job description is provided, set
   job_match_score to null.
3. Identify missing keywords the resume should include, based on the job description if
   provided, or the candidate's apparent target role if not.
4. Identify the 3-5 weakest bullet points in the resume and rewrite them using the XYZ
   formula (Accomplished X, measured by Y, by doing Z). If a metric is not present in the
   original, write a realistic placeholder in brackets like [X%] rather than inventing a
   false specific number.
5. If a job description is provided, also produce a full tailored rewrite of the resume
   optimized for that role, preserving truthful content but improving structure, keyword
   coverage, and bullet phrasing. If no job description is provided, set tailored_resume
   to null.

Reference examples of bullet quality:
Bad bullet: "Responsible for managing team projects"
Good bullet: "Led a 6-person engineering team to ship a payments feature 3 weeks ahead of
schedule, reducing checkout latency by 40%"

Bad bullet: "Worked on backend systems"
Good bullet: "Re-architected a monolithic billing service into microservices, cutting
deployment time from 45 to 6 minutes"

Always return valid JSON matching this exact schema. Never include commentary, markdown
formatting, or code fences outside the JSON object:
{
  "ats_score": 0,
  "job_match_score": 0,
  "missing_keywords": [],
  "weak_bullets": [{"original": "", "rewritten": "", "reason": ""}],
  "summary_feedback": "",
  "tailored_resume": null
}"""


# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------

def extract_text(uploaded_file) -> str:
    """Extract plain text from an uploaded .pdf, .docx, or .txt file."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        import pdfplumber
        text_parts = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    if name.endswith(".docx"):
        import docx
        d = docx.Document(uploaded_file)
        return "\n".join(p.text for p in d.paragraphs)

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


# --------------------------------------------------------------------------
# Gemini call
# --------------------------------------------------------------------------

def get_api_key() -> str:
    # Streamlit Cloud: st.secrets. Local dev: environment variable.
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY", "")


def analyze_resume(resume_text: str, jd_text: str | None) -> dict:
    client = genai.Client(api_key=get_api_key())

    user_message = f"RESUME:\n{resume_text}\n\n"
    if jd_text:
        user_message += f"JOB DESCRIPTION:\n{jd_text}"
    else:
        user_message += "No job description provided -- perform a general ATS and quality scan only."

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"

    response = client.models.generate_content(
        model=MODEL,
        contents=full_prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = response.text
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\n\nRaw output:\n{raw_text}") from e


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Resume Optimizer AI", page_icon="📄", layout="wide")

st.title("📄 Resume Optimizer AI")
st.caption("ATS scoring, job-match scoring, and bullet rewriting -- powered by a single structured prompt.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Resume")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    pasted_resume = st.text_area("...or paste resume text", height=200, placeholder="Paste resume text here instead of uploading")

with col2:
    st.subheader("2. Target role (optional)")
    jd_text = st.text_area("Paste the job description", height=260, placeholder="Leave blank for a general ATS scan")

analyze_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if analyze_clicked:
    if not get_api_key():
        st.error("No GEMINI_API_KEY found. Set it in Streamlit secrets (deployed) or as an environment variable (local).")
        st.stop()

    resume_text = ""
    if uploaded_file is not None:
        try:
            resume_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.stop()
    elif pasted_resume.strip():
        resume_text = pasted_resume.strip()

    if not resume_text:
        st.warning("Add a resume first -- upload a file or paste the text.")
        st.stop()

    with st.spinner("Running ATS scan and recruiter review..."):
        try:
            result = analyze_resume(resume_text, jd_text.strip() or None)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.divider()

    # Scores
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        st.metric("ATS Compatibility", f"{result.get('ats_score', 0)}/100")
        st.progress(result.get("ats_score", 0) / 100)
    with score_col2:
        match_score = result.get("job_match_score")
        if match_score is None:
            st.metric("Job Match", "— (no JD)")
        else:
            st.metric("Job Match", f"{match_score}/100")
            st.progress(match_score / 100)

    # Summary
    st.subheader("Recruiter Summary")
    st.write(result.get("summary_feedback", ""))

    # Missing keywords
    st.subheader("Missing Keywords")
    keywords = result.get("missing_keywords", [])
    if keywords:
        st.markdown(" ".join(f"`{kw}`" for kw in keywords))
    else:
        st.success("No critical gaps found.")

    # Bullet rewrites
    weak_bullets = result.get("weak_bullets", [])
    if weak_bullets:
        st.subheader("Bullet Rewrites")
        for b in weak_bullets:
            st.markdown(f"~~{b.get('original', '')}~~")
            st.markdown(f"**→ {b.get('rewritten', '')}**")
            st.caption(b.get("reason", ""))
            st.divider()

    # Tailored rewrite
    tailored = result.get("tailored_resume")
    if tailored:
        st.subheader("Tailored Resume Rewrite")
        st.text_area("Full rewrite", value=tailored, height=500)
        st.download_button("Download as .txt", data=tailored, file_name="tailored_resume.txt")
