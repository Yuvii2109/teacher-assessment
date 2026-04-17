import requests
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import os
import re
import json
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# --- Load Environment Variables ---
load_dotenv()

# --- Configure Page ---
st.set_page_config(page_title="Webinar Insights Dashboard", layout="wide")
st.title("Teacher Mental Health Training: Insights Dashboard")

# --- Backend Config Setup ---
st.sidebar.header("Backend Configuration Status")
api_key = os.getenv("GEMINI_API_KEY")
pre_url = os.getenv("PRE_SHEET_URL")
post_url = os.getenv("POST_SHEET_URL")

if api_key:
    st.sidebar.success("API Key loaded securely")
else:
    st.sidebar.error("GEMINI_API_KEY missing from .env")

if pre_url and post_url:
    st.sidebar.success("Google Sheets Links configured")
else:
    st.sidebar.error("PRE_SHEET_URL or POST_SHEET_URL missing from .env")

# --- Base Answer Keys (English Concepts) ---
PRE_ASSESSMENT_KEY = {
    "Mental health is best understood as:": "emotional, social, and psychological well-being",
    "Which is a common source of stress for students today?": "academic demands, peer pressure, and family expectations",
    "Which is the clearest early warning sign a teacher may observe?": "showing sudden withdrawal over several days",
    "A student who is usually cheerful has become quiet, avoids friends, and has stopped submitting work. What should the teacher do first?": "watch the pattern and speak privately",
    "Which classroom practice is most likely to improve emotional safety?": "using respectful language and encouragement",
    "Before a test, a student says, \"I know I will not do well.\" What is the most helpful immediate teacher response?": "guide the student to begin with familiar questions",
    "Which approach is most appropriate while talking to parents about a concern?": "sharing observations and inviting partnership",
    "Which action should a teacher avoid when concerned about a student?": "deciding on a diagnosis from classroom signs",
    "Which assessment practice is most likely to reduce student stress?": "offering clear success criteria in advance",
    "One student is irritable, one is withdrawn, and one frequently reports headaches before tests. What is the best interpretation?": "they are showing possible signs of stress",
    "A teacher notices repeated emotional distress even after classroom support. What is the best next step?": "move the concern through school support channels",
    "Which statement best reflects a mentally healthy school culture?": "academic progress depends on emotional safety"
}

POST_ASSESSMENT_KEY = {
    "Good mental health in students is best reflected when they:": "manage emotions and function reasonably well",
    "Which description best matches burnout?": "exhaustion, detachment, and reduced motivation",
    "Which factor is most closely linked to healthy student development in school?": "consistent adult support and connection",
    "A student who usually participates well has stopped answering and avoids eye contact. What is the most appropriate first response?": "observe carefully and check in privately",
    "Which classroom practice best supports student emotional safety?": "acknowledging effort in a respectful way",
    "A teacher wants to speak to parents about a student's recent change in behaviour. Which opening is best?": "We have noticed some changes and want to support together.",
    "Which is the best example of a healthy teacher response to student stress?": "acknowledging the feeling and offering calm guidance",
    "Which assessment practice best supports student well-being?": "giving clear criteria and calm instructions",
    "A teacher notices one student becomes restless before tests, one becomes silent during group work, and one often says, \"I cannot do this.\" What is the best next move?": "respond to each pattern with appropriate support",
    "A school wants to become more mentally healthy. Which change is likely to have the strongest everyday effect?": "combining supportive teaching with referral systems",
    "A teacher has supported a student in class, checked in privately, and still sees persistent distress. What should the teacher conclude?": "the concern may need referral support",
    "Which statement best reflects the spirit of the workshop?": "teachers support mental health through daily practice"
}

# --- Helper Functions ---
def normalize_string(s):
    """Removes all non-alphanumeric characters for bulletproof column matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()

def get_matched_column(df, question):
    """Finds the correct column even if punctuation or numbering changes."""
    q_clean = normalize_string(question)[:25]
    for col in df.columns:
        if q_clean in normalize_string(col):
            return col
    return None

@st.cache_data(ttl=300)
def fetch_google_sheet_data(url):
    try:
        if "export?format=csv" not in url:
            match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
            if match:
                doc_id = match.group(1)
                url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
                
        # Fetch the data over HTTP to prevent Windows file path errors
        response = requests.get(url)
        response.raise_for_status() # Fails safely if the URL is broken or private
        
        # Load the text content directly into Pandas from memory
        return pd.read_csv(io.StringIO(response.text))
        
    except Exception as e:
        st.error(f"Failed to fetch data from the provided URL. Error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def build_dynamic_answer_mapping(df, base_answer_key, current_api_key):
    """Uses Gemini to identify correct answers across all questions in a single batched call."""
    client = genai.Client(api_key=current_api_key)
    dynamic_key = {}
    payload_to_grade = {}

    # 1. Aggregate all questions and unique responses locally
    for question, correct_concept in base_answer_key.items():
        matched_col = get_matched_column(df, question)
        if not matched_col:
            dynamic_key[question] = [correct_concept]
            continue
            
        unique_responses = []
        for val in df[matched_col].dropna().astype(str):
            for v in val.split(','):
                v_clean = v.strip()
                if v_clean and v_clean not in unique_responses:
                    unique_responses.append(v_clean)
                    
        payload_to_grade[question] = {
            "correct_concept": correct_concept,
            "user_responses": unique_responses
        }

    # 2. Make ONE API call for the entire dataset
    prompt = f"""
    You are a survey grading assistant.
    Below is a JSON object mapping questions to their "correct_concept" in English, along with a list of raw "user_responses" (which may include Gujarati or typos).
    
    {json.dumps(payload_to_grade, ensure_ascii=False)}
    
    For each question, identify ALL "user_responses" that semantically mean the same thing as the "correct_concept".
    
    Return ONLY a raw JSON object where the keys are the exact questions, and the values are flat arrays of the approved response strings. Do not include markdown blocks like ```json.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        text = response.text.strip()
        
        # Aggressively extract the JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
            
        approved_answers_dict = json.loads(text)
        
        # 3. Rebuild the dynamic key from the single LLM response
        for question, correct_concept in base_answer_key.items():
            if question in approved_answers_dict:
                approved = approved_answers_dict[question]
                if isinstance(approved, list):
                    dynamic_key[question] = [correct_concept] + approved
                else:
                    dynamic_key[question] = [correct_concept]
            elif question not in dynamic_key: # Fallback if LLM missed a key
                dynamic_key[question] = [correct_concept]
                
    except Exception as e:
        print(f"Error parsing batched Gemini mapping: {e}")
        # Fallback to base key if the batch fails entirely
        for question, correct_concept in base_answer_key.items():
            if question not in dynamic_key:
                dynamic_key[question] = [correct_concept]
                
    return dynamic_key

def apply_grid(fig):
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinecolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinecolor='LightGray')
    )
    return fig

def grade_questions(df, dynamic_answer_key):
    results = {}
    total_responses = len(df)
    for question, correct_answers in dynamic_answer_key.items():
        matched_col = get_matched_column(df, question)
        if matched_col:
            correct_count = df[matched_col].astype(str).apply(
                lambda x: 1 if any(ans.lower() in x.lower() for ans in correct_answers) else 0
            ).sum()
            accuracy = (correct_count / total_responses) * 100 if total_responses > 0 else 0
            results[question] = accuracy
    return results

def get_participant_scores(df, dynamic_answer_key):
    scores = []
    for index, row in df.iterrows():
        correct = 0
        for question, correct_answers in dynamic_answer_key.items():
            matched_col = get_matched_column(df, question)
            if matched_col and pd.notna(row[matched_col]):
                if any(ans.lower() in str(row[matched_col]).lower() for ans in correct_answers):
                    correct += 1
        scores.append(correct)
    return scores

def generate_gemini_insights(pre_data, post_data, current_api_key):
    client = genai.Client(api_key=current_api_key)
    prompt = """
    You are a Senior AI Data Analyst evaluating a teacher mental health training program. 
    Your task is to generate a comprehensive, deep-dive insight report based on the provided assessment accuracy data. 
    
    CRITICAL FORMATTING RULES:
    1. The report must be highly skimmable. 
    2. Rely heavily on structured bullet points, sub-bullets, and bold keywords. 
    3. DO NOT write long, dense paragraphs. 
    4. DO NOT use any emojis or informal language. Maintain a strictly professional, corporate tone.
    
    Format your response EXACTLY using these four markdown headers:
    
    ### Attendee-Facing Highlights (Public)
    (Provide a robust summary of the group's collective success. Detail specific areas where the teachers demonstrated strong understanding or significant improvement.)
    
    ### Presenter Internal Record: Data Trajectory (Private)
    (Provide a detailed breakdown of the data shift. Analyze the overall baseline versus post-webinar averages. Highlight the top questions that saw the highest accuracy.)
    
    ### Deep Dive: Critical Knowledge Gaps (Private)
    (Identify the lowest-performing concepts or areas of regression. Provide an analytical hypothesis on WHY teachers might still be struggling with these specific concepts.)
    
    ### Strategic Action Plan (Private)
    (For the gaps identified above, provide a concrete, multi-step intervention strategy including specific talking points for follow-ups.)
    
    DATA TO ANALYZE:
    """
    if pre_data and post_data:
        prompt += f"\nPre-Webinar Accuracy:\n{pre_data}\n\nPost-Webinar Accuracy:\n{post_data}"
    
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def create_pdf(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Teacher Mental Health Training - Insights Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(10)
    
    replacements = {'**': '', '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2013': "-", '\u2014': "-", '\u2022': "-", '*': '-', '\t': ' '}
    clean_text = report_text
    for old, new in replacements.items():
        clean_text = clean_text.replace(old, new)
        
    clean_text = clean_text.encode('latin-1', 'ignore').decode('latin-1')

    pdf.set_font("helvetica", "", 12)
    for line in clean_text.split('\n'):
        line = line.strip()
        if not line or set(line) <= {'-', '_', ' '}:
            pdf.ln(4)
            continue
        if line.startswith('### '):
            pdf.ln(4)
            pdf.set_font("helvetica", "B", 14)
            pdf.write(10, line.replace('### ', '') + '\n')
            pdf.set_font("helvetica", "", 12)
        else:
            pdf.write(8, line + '\n')
    return bytes(pdf.output())

# --- Dashboard UI ---
st.markdown("Click the button below to fetch the latest responses directly from the live Google Sheets and analyze the performance.")
process_btn = st.button("Fetch Live Data & Generate Dashboard", width="stretch", type="primary")

if process_btn:
    if not pre_url or not post_url:
        st.warning("Please configure the PRE_SHEET_URL and POST_SHEET_URL in your .env file.")
    else:
        st.session_state['process_clicked'] = True

if st.session_state.get('process_clicked', False) and pre_url and post_url:
    with st.spinner("Fetching data from Google Sheets..."):
        pre_df = fetch_google_sheet_data(pre_url)
        post_df = fetch_google_sheet_data(post_url)
        
    if pre_df is not None and post_df is not None:
        with st.spinner("Building dynamic multi-lingual grading keys (this takes 10-15 seconds on the first run)..."):
            dynamic_pre_key = build_dynamic_answer_mapping(pre_df, PRE_ASSESSMENT_KEY, api_key)
            dynamic_post_key = build_dynamic_answer_mapping(post_df, POST_ASSESSMENT_KEY, api_key)

        pre_results = grade_questions(pre_df, dynamic_pre_key)
        post_results = grade_questions(post_df, dynamic_post_key)
        
        pre_scores = get_participant_scores(pre_df, dynamic_pre_key)
        post_scores = get_participant_scores(post_df, dynamic_post_key)
        
        df_pre = pd.DataFrame(list(pre_results.items()), columns=['Question', 'Accuracy (%)'])
        df_pre['Question_Short'] = [f"Q{i+1}" for i in range(len(df_pre))]
        
        df_post = pd.DataFrame(list(post_results.items()), columns=['Question', 'Accuracy (%)'])
        df_post['Question_Short'] = [f"Q{i+1}" for i in range(len(df_post))]

        df_delta = pd.DataFrame({
            'Question': df_pre['Question'],
            'Question_Short': df_pre['Question_Short'],
            'Improvement (%)': df_post['Accuracy (%)'] - df_pre['Accuracy (%)']
        })

        st.header("Aggregate Performance Metrics")
        
        # --- Section 1: Executive Participant Overview ---
        st.subheader("Participant Overview")
        col_pre, col_post = st.columns(2)
        with col_pre:
            st.markdown("**Pre-Webinar Statistics**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Participants", len(pre_scores))
            c2.metric("Avg Score", f"{sum(pre_scores)/len(pre_scores):.1f} / 12" if pre_scores else "0.0 / 12")
            c3.metric("Highest Score", f"{max(pre_scores)} / 12" if pre_scores else "0 / 12")
        with col_post:
            st.markdown("**Post-Webinar Statistics**")
            c4, c5, c6 = st.columns(3)
            c4.metric("Total Participants", len(post_scores))
            
            pre_avg_score = sum(pre_scores)/len(pre_scores) if pre_scores else 0
            post_avg_score = sum(post_scores)/len(post_scores) if post_scores else 0
            
            c5.metric("Avg Score", f"{post_avg_score:.1f} / 12", delta=f"{(post_avg_score - pre_avg_score):.1f}")
            highest_pre = max(pre_scores) if pre_scores else 0
            highest_post = max(post_scores) if post_scores else 0
            c6.metric("Highest Score", f"{highest_post} / 12", delta=f"{highest_post - highest_pre}")

        st.markdown("---")

        # --- Section 2: Question Level Improvement ---
        st.subheader("Webinar Impact: Knowledge Growth per Question")
        st.info("Positive values (Green) indicate knowledge gained post-webinar. Negative values (Red) indicate areas of confusion or regression.")
        
        fig_delta = px.bar(df_delta, x="Question_Short", y="Improvement (%)", 
                           title="Improvement Post-Webinar (Delta %)",
                           hover_data=['Question'],
                           color="Improvement (%)", color_continuous_scale=px.colors.diverging.RdYlGn,
                           color_continuous_midpoint=0)
        fig_delta.update_traces(marker_line_color='black', marker_line_width=1)
        st.plotly_chart(apply_grid(fig_delta), width="stretch")

        # --- Section 3: Question Level Accuracy (Pre vs Post) ---
        st.markdown("---")
        st.subheader("Detailed Pre vs Post Accuracy Comparison")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=df_pre['Question_Short'], y=df_pre['Accuracy (%)'], name='Pre-Webinar', marker_color='#1f77b4', marker_line_color='black', marker_line_width=1, hovertext=df_pre['Question']))
        fig_comp.add_trace(go.Bar(x=df_post['Question_Short'], y=df_post['Accuracy (%)'], name='Post-Webinar', marker_color='#2ca02c', marker_line_color='black', marker_line_width=1, hovertext=df_post['Question']))
        fig_comp.update_layout(title="Pre vs Post Webinar Comparison per Question", barmode='group', yaxis_range=[0, 100])
        st.plotly_chart(apply_grid(fig_comp), width="stretch")

        # --- Section 4: Participant Score Distribution ---
        st.markdown("---")
        st.subheader("Participant Score Distribution (Frequency)")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=pre_scores, name='Pre-Webinar', marker_color='#1f77b4', opacity=0.75, xbins=dict(start=-0.5, end=12.5, size=1), marker_line_color='black', marker_line_width=1))
        fig_dist.add_trace(go.Histogram(x=post_scores, name='Post-Webinar', marker_color='#2ca02c', opacity=0.75, xbins=dict(start=-0.5, end=12.5, size=1), marker_line_color='black', marker_line_width=1))
        fig_dist.update_layout(barmode='overlay', title="Frequency of Participant Scores (Out of 12)", xaxis_title="Score (out of 12)", yaxis_title="Number of Teachers", xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(apply_grid(fig_dist), width="stretch")

        # --- Section 5: AI Insight Generation & Export ---
        st.markdown("---")
        st.header("Actionable Insights")
        
        if not api_key:
            st.warning("Please verify your GEMINI_API_KEY in the .env file to generate AI insights.")
        else:
            if "generated_insights" not in st.session_state:
                with st.spinner("Analyzing data and building action plan..."):
                    insights = generate_gemini_insights(pre_results, post_results, api_key)
                    st.session_state["generated_insights"] = insights
                    
            if "generated_insights" in st.session_state:
                st.markdown(st.session_state["generated_insights"])
                pdf_bytes = create_pdf(st.session_state["generated_insights"])
                st.download_button(label="Download Insights Report (PDF)", data=pdf_bytes, file_name="Webinar_Insights_Report.pdf", mime="application/pdf", width="stretch")