"""
Resume Optimizer AI
--------------------
ATS scoring, job-match scoring, bullet rewriting, and tailored resume
rewriting -- driven by a single structured prompt sent to Claude.

Usage:
    python resume_optimizer.py --resume resume.pdf
    python resume_optimizer.py --resume resume.docx --jd job_description.txt
    python resume_optimizer.py --resume resume.txt --jd job_description.txt --json out.json

Requires:
    pip install google-genai pdfplumber python-docx rich
    export GEMINI_API_KEY=...   (free key from https://aistudio.google.com/apikey)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from google import genai

console = Console()

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

def extract_text(path: Path) -> str:
    """Extract plain text from a .pdf, .docx, or .txt resume file."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    if suffix == ".docx":
        import docx
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class WeakBullet:
    original: str
    rewritten: str
    reason: str


@dataclass
class AnalysisResult:
    ats_score: int
    job_match_score: Optional[int]
    missing_keywords: list[str] = field(default_factory=list)
    weak_bullets: list[WeakBullet] = field(default_factory=list)
    summary_feedback: str = ""
    tailored_resume: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisResult":
        return cls(
            ats_score=d.get("ats_score", 0),
            job_match_score=d.get("job_match_score"),
            missing_keywords=d.get("missing_keywords", []),
            weak_bullets=[WeakBullet(**b) for b in d.get("weak_bullets", [])],
            summary_feedback=d.get("summary_feedback", ""),
            tailored_resume=d.get("tailored_resume"),
        )


# --------------------------------------------------------------------------
# Claude call
# --------------------------------------------------------------------------

def analyze_resume(resume_text: str, jd_text: Optional[str]) -> AnalysisResult:
    client = genai.Client()  # reads GEMINI_API_KEY from env

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
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Model did not return valid JSON: {e}\n\nRaw output:\n{raw_text}"
        ) from e

    return AnalysisResult.from_dict(parsed)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def score_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 60:
        return "yellow"
    return "red"


def render_result(result: AnalysisResult) -> None:
    console.print()

    # Score panels
    score_table = Table.grid(expand=True)
    score_table.add_column(ratio=1)
    score_table.add_column(ratio=1)

    ats_color = score_color(result.ats_score)
    ats_panel = Panel(
        Text(f"{result.ats_score}/100", style=f"bold {ats_color}", justify="center"),
        title="ATS Compatibility",
        border_style=ats_color,
    )

    if result.job_match_score is None:
        match_panel = Panel(
            Text("— (no JD provided)", style="dim", justify="center"),
            title="Job Match",
            border_style="grey50",
        )
    else:
        match_color = score_color(result.job_match_score)
        match_panel = Panel(
            Text(f"{result.job_match_score}/100", style=f"bold {match_color}", justify="center"),
            title="Job Match",
            border_style=match_color,
        )

    score_table.add_row(ats_panel, match_panel)
    console.print(score_table)

    # Summary
    console.print(Panel(result.summary_feedback, title="Recruiter Summary", border_style="blue"))

    # Missing keywords
    if result.missing_keywords:
        kw_text = "  ".join(f"[bold red]· {kw}[/bold red]" for kw in result.missing_keywords)
        console.print(Panel(kw_text, title="Missing Keywords", border_style="red"))
    else:
        console.print(Panel("No critical gaps found.", title="Missing Keywords", border_style="green"))

    # Bullet rewrites
    if result.weak_bullets:
        bullet_table = Table(title="Bullet Rewrites", show_lines=True, expand=True)
        bullet_table.add_column("Original", style="strike red", ratio=1)
        bullet_table.add_column("Rewritten", style="green", ratio=1)
        bullet_table.add_column("Why", style="dim", ratio=1)
        for b in result.weak_bullets:
            bullet_table.add_row(b.original, b.rewritten, b.reason)
        console.print(bullet_table)

    # Tailored rewrite
    if result.tailored_resume:
        console.print(Panel(result.tailored_resume, title="Tailored Resume Rewrite", border_style="magenta"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resume Optimizer AI -- ATS scoring and rewriting")
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf, .docx, .txt)")
    parser.add_argument("--jd", help="Path to job description text file (optional)")
    parser.add_argument("--json", dest="json_out", help="Optional path to also save raw JSON results")
    args = parser.parse_args()

    resume_path = Path(args.resume)
    if not resume_path.exists():
        console.print(f"[bold red]Error:[/bold red] resume file not found: {resume_path}")
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        console.print("[bold red]Error:[/bold red] set GEMINI_API_KEY in your environment first.")
        console.print("[dim]Get a free key at https://aistudio.google.com/apikey[/dim]")
        sys.exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Parsing resume...", total=None)
        resume_text = extract_text(resume_path)

        jd_text = None
        if args.jd:
            jd_path = Path(args.jd)
            if not jd_path.exists():
                console.print(f"[bold red]Error:[/bold red] job description file not found: {jd_path}")
                sys.exit(1)
            jd_text = jd_path.read_text(encoding="utf-8")

        progress.update(task, description="Running ATS scan and recruiter review...")
        result = analyze_resume(resume_text, jd_text)
        progress.update(task, description="Done.")

    render_result(result)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, indent=2, default=lambda o: o.__dict__)
        console.print(f"\n[dim]Raw JSON saved to {args.json_out}[/dim]")


if __name__ == "__main__":
    main()
