# Resume Optimizer AI

**A structured-output prompt engineering project: an LLM system prompt that turns a single API call into a reliable ATS scorer, recruiter critic, and resume rewriter.**

Live demo: https://shivank-resume-optimiser.streamlit.app/

---

## Overview

Resume Optimizer AI takes a resume (PDF, DOCX, or plain text) and an optional job description, and returns:

- An **ATS compatibility score** (0–100), simulating how an applicant tracking system parses and ranks the document
- A **job match score** (0–100) when a job description is provided
- A list of **missing keywords** the resume should include
- **Rewritten bullet points** for the weakest lines, using a fixed accomplishment formula
- An optional **full tailored resume rewrite** targeted at the job description

The point of the project isn't the file parsing or the UI — both are intentionally thin. The point is the prompt: one system prompt does role-setting, few-shot calibration, conditional logic, anti-hallucination guardrails, and strict structured output, all in a single non-conversational call.

---

## The prompt engineering

This is the core artifact of the project. The system prompt is designed around five techniques:

### 1. Role-based prompting
The model is cast as two evaluators at once — *"an ATS simulator combined with a senior FAANG technical recruiter with 15 years of hiring experience."* This dual framing matters: an ATS evaluates formatting and keyword density mechanically, while a recruiter evaluates substance and impact. Asking for both in one persona produces scores that account for both axes instead of just one.

### 2. Few-shot calibration
Rather than asking the model to "improve weak bullets" and trusting its judgment of what "weak" means, the prompt embeds two concrete bad/good pairs:

```
Bad:  "Responsible for managing team projects"
Good: "Led a 6-person engineering team to ship a payments feature
       3 weeks ahead of schedule, reducing checkout latency by 40%"
```

This anchors the model to a specific bar — the XYZ formula (accomplished X, measured by Y, by doing Z) — instead of a vague notion of "better."

### 3. Forced structured output
The prompt specifies the exact JSON schema inline and explicitly forbids commentary or markdown fences outside it. The application code treats this as a contract, not a hope: it strips any stray code fences and parses with `json.loads`, catching `JSONDecodeError` and surfacing the raw model output for debugging rather than failing silently.

### 4. Conditional schema logic
`job_match_score` and `tailored_resume` are explicitly instructed to be `null` when no job description is supplied. This prevents the model from inventing a comparison score against nothing, or hallucinating a "tailored" rewrite with no target to tailor toward — a failure mode that's easy to miss if you only test the happy path.

### 5. Anti-fabrication constraint
The instruction to rewrite weak bullets includes a specific guardrail: if the original resume has no metric, the model must insert a bracketed placeholder like `[X%]` rather than inventing a specific, false number. This is the difference between a tool that helps someone write a stronger bullet and a tool that quietly puts fabricated statistics on someone's resume.

---

## System prompt (excerpt)

```
You are an ATS (Applicant Tracking System) simulator combined with a senior
FAANG technical recruiter with 15 years of hiring experience. You evaluate
resumes the way both a parsing algorithm and a human reviewer would.

Your tasks:
1. Score ATS compatibility (0-100) based on formatting, keyword density,
   section structure, and parseability.
2. If a job description is provided, score job match (0-100)...
   If no job description is provided, set job_match_score to null.
3. Identify missing keywords...
4. Rewrite the 3-5 weakest bullet points using the XYZ formula...
   If a metric is not present in the original, write a realistic
   placeholder in brackets like [X%] rather than inventing a false number.
5. If a job description is provided, also produce a full tailored rewrite...

Always return valid JSON matching this exact schema. Never include
commentary, markdown formatting, or code fences outside the JSON object.
```

Full prompt is in `app.py` / `resume_optimizer.py`.

---

## Architecture

```
Input (PDF / DOCX / text)
        │
        ▼
  Text extraction (pdfplumber / python-docx)
        │
        ▼
  Single structured prompt → Gemini 2.5 Flash
        │
        ▼
  Defensive JSON parsing (fence-stripping, error surfacing)
        │
        ▼
  Rendered scorecard (Streamlit UI / Rich terminal UI)
```

Two interchangeable front ends share the same prompt and parsing logic:

- `app.py` — Streamlit web app, deployable for free, used for the live demo
- `resume_optimizer.py` — CLI version, same logic, for local/terminal use

---

## Stack

- **Model**: Gemini 2.5 Flash (free tier, 1,500 requests/day)
- **Parsing**: `pdfplumber`, `python-docx`
- **Interface**: Streamlit (web), `rich` (CLI)
- **Output validation**: strict JSON schema enforced via prompt + defensive parsing

---

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...   # free key: https://aistudio.google.com/apikey
streamlit run app.py
```

CLI alternative:

```bash
python resume_optimizer.py --resume resume.pdf --jd job_description.txt
```

---

## What I'd improve next

- Schema validation with `pydantic` and an automatic retry if the model ever returns malformed JSON
- A small eval set of resumes with known weak points, to measure whether prompt edits actually improve scoring consistency
- Batch mode: score multiple resumes against one job description and rank them
