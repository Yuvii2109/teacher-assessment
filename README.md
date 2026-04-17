# Teacher Mental Health Training Insights Dashboard

A Streamlit-based dashboard for analyzing pre-webinar and post-webinar teacher assessment data, visualizing learning outcomes, and generating AI-assisted insight reports.

## Overview

This project helps facilitators evaluate teacher mental health training impact by:

- Uploading pre-assessment and post-assessment CSV files
- Computing question-level accuracy and participant-level scores
- Visualizing performance trends and score distributions
- Generating structured AI insights using Gemini
- Exporting insights as a PDF report

The main application entry point is `app.py`.

## Features

- Dual CSV upload workflow (pre and post webinar)
- Question-level accuracy analysis
- Average score comparison (pre vs post)
- Participant score frequency histograms
- AI-generated strategic insights with fixed reporting sections
- One-click PDF export of generated insights

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Google GenAI SDK (`google-genai`)
- python-dotenv
- fpdf2

## Project Structure

```
teacher-assessment/
├── app.py
├── requirements.txt
└── .gitignore
```

## Prerequisites

- Python 3.9+
- A Gemini API key

## Installation

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

If the API key is not found, the dashboard still runs analytics and charts, but AI insight generation is disabled.

## Running the App

```bash
streamlit run app.py
```

By default, Streamlit opens a local URL (typically `http://localhost:8501`).

## Input Data Expectations

- Input files must be CSV.
- You can upload:
  - only pre-assessment,
  - only post-assessment, or
  - both for direct comparison.
- Column matching is heuristic-based:
  - For each expected question, the app searches for a CSV column containing the first 20 characters of that question (case-insensitive).
- Answer scoring is substring-based:
  - A response is treated as correct if the expected answer text appears within the participant response (case-insensitive).

## How Scoring Works

- `grade_questions(df, answer_key)`
  - Computes per-question accuracy percentage.
- `get_participant_scores(df, answer_key)`
  - Computes each participant's overall percentage score.

The app includes separate hardcoded answer keys for pre-assessment and post-assessment questions.

## AI Insight Generation

When `GEMINI_API_KEY` is available, the app sends aggregated accuracy data to Gemini (`gemini-2.5-flash`) and requests a structured report with these sections:

1. Attendee-Facing Highlights (Public)
2. Presenter Internal Record: Data Trajectory (Private)
3. Deep Dive: Critical Knowledge Gaps (Private)
4. Strategic Action Plan (Private)

## PDF Export

Generated AI insights can be downloaded as `Webinar_Insights_Report.pdf`.

The app sanitizes markdown/unicode content before PDF rendering to avoid font and encoding issues.

## Notes and Limitations

- Column matching depends on question text similarity; renamed headers may reduce match quality.
- Substring answer matching is permissive and may count partially matching text as correct.
- API/network issues can prevent insight generation; the app returns the exception message in the UI.

## Security

- Keep your `.env` file private.
- `.gitignore` excludes `.env`, `venv/`, `__pycache__/`, and `*.pyc`.

## Quick Start Summary

```bash
pip install -r requirements.txt
# create .env with GEMINI_API_KEY
streamlit run app.py
```
