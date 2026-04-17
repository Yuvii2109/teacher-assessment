import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import os
from dotenv import load_dotenv
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# --- Load Environment Variables ---
load_dotenv()

# --- Configure Page ---
st.set_page_config(page_title="Webinar Insights Dashboard", layout="wide")
st.title("Teacher Mental Health Training: Insights Dashboard")

# --- API Key Setup ---
st.sidebar.header("Configuration")
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    st.sidebar.success("API Key loaded securely from .env file")
else:
    st.sidebar.error("GEMINI_API_KEY not found. Please check your .env file.")

# --- Answer Keys ---
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
    "1. Good mental health in students is best reflected when they:": "manage emotions and function reasonably well",
    "2. Which description best matches burnout?": "exhaustion, detachment, and reduced motivation",
    "3. Which factor is most closely linked to healthy student development in school?": "consistent adult support and connection",
    "4. A student who usually participates well has stopped answering and avoids eye contact. What is the most appropriate first response?": "observe carefully and check in privately",
    "5. Which classroom practice best supports student emotional safety?": "acknowledging effort in a respectful way",
    "6. A teacher wants to speak to parents about a student's recent change in behaviour. Which opening is best?": "We have noticed some changes and want to support together.",
    "7. Which is the best example of a healthy teacher response to student stress?": "acknowledging the feeling and offering calm guidance",
    "8. Which assessment practice best supports student well-being?": "giving clear criteria and calm instructions",
    "9. A teacher notices one student becomes restless before tests, one becomes silent during group work, and one often says, \"I cannot do this.\" What is the best next move?": "respond to each pattern with appropriate support",
    "10. A school wants to become more mentally healthy. Which change is likely to have the strongest everyday effect?": "combining supportive teaching with referral systems",
    "11. A teacher has supported a student in class, checked in privately, and still sees persistent distress. What should the teacher conclude?": "the concern may need referral support",
    "12. Which statement best reflects the spirit of the workshop?": "teachers support mental health through daily practice"
}

# --- Helper Functions ---
def apply_grid(fig):
    """Applies a clean background and explicit gridlines to Plotly figures."""
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinecolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinecolor='LightGray')
    )
    return fig

def grade_questions(df, answer_key):
    results = {}
    total_responses = len(df)
    
    for question, correct_answer in answer_key.items():
        matched_col = next((col for col in df.columns if question.lower()[:20] in col.lower()), None)
        if matched_col:
            correct_count = df[matched_col].astype(str).apply(
                lambda x: 1 if correct_answer.lower() in x.lower() else 0
            ).sum()
            accuracy = (correct_count / total_responses) * 100 if total_responses > 0 else 0
            results[question] = accuracy
    return results

def get_participant_scores(df, answer_key):
    scores = []
    
    for index, row in df.iterrows():
        correct = 0
        for question, correct_answer in answer_key.items():
            matched_col = next((col for col in df.columns if question.lower()[:20] in col.lower()), None)
            if matched_col and pd.notna(row[matched_col]):
                if correct_answer.lower() in str(row[matched_col]).lower():
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
    (Provide a robust, celebratory summary of the group's collective success. Detail 4 to 5 specific areas where the teachers demonstrated strong understanding or significant improvement. Explain briefly why these specific competencies are crucial for classroom success. Tone: Validating and encouraging.)
    
    ### Presenter Internal Record: Data Trajectory (Private)
    (Provide a detailed breakdown of the data shift. Analyze the overall baseline versus post-webinar averages. Specifically highlight the top 2 questions that saw the highest accuracy and analyze what this indicates about the effectiveness of the training delivery.)
    
    ### Deep Dive: Critical Knowledge Gaps (Private)
    (Identify the top 3 lowest-performing concepts or areas of regression in the post-webinar data. Do not just state the data; provide an analytical hypothesis on WHY teachers might still be struggling with these specific concepts based on common pedagogical or psychological challenges.)
    
    ### Strategic Action Plan (Private)
    (For each of the 3 gaps identified above, provide a concrete, multi-step intervention strategy. For each gap include:
    - The specific gap.
    - A specific talking point for a follow-up email.
    - A 5-minute micro-learning topic or exercise to integrate into the next staff meeting or webinar.)
    
    DATA TO ANALYZE:
    """
    
    if pre_data and post_data:
        prompt += f"\nPre-Webinar Accuracy:\n{pre_data}\n\nPost-Webinar Accuracy:\n{post_data}"
    elif pre_data:
        prompt += f"\nPre-Webinar Accuracy (Baseline only):\n{pre_data}"
    elif post_data:
        prompt += f"\nPost-Webinar Accuracy (Post-training only):\n{post_data}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
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
    
    replacements = {
        '**': '', 
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': "-", '\u2014': "-",
        '\u2022': "-",                
        '*': '-',                     
        '\t': '    '  
    }
    
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
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Pre-Webinar Data")
    pre_file = st.file_uploader("Upload Pre-Assessment CSV", type=['csv'], key="pre")

with col2:
    st.subheader("Upload Post-Webinar Data")
    post_file = st.file_uploader("Upload Post-Assessment CSV", type=['csv'], key="post")

st.markdown("---")
process_btn = st.button("Process Data & Generate Dashboard", width="stretch")

if process_btn:
    if not pre_file and not post_file:
        st.warning("Please upload at least one CSV file (Pre or Post) to process.")
    else:
        st.session_state['process_clicked'] = True

if st.session_state.get('process_clicked', False) and (pre_file or post_file):
    pre_results, post_results = None, None
    df_pre, df_post = None, None
    pre_scores, post_scores = [], []
    
    st.header("Aggregate Performance Metrics")
    
    if pre_file:
        pre_df = pd.read_csv(pre_file)
        pre_results = grade_questions(pre_df, PRE_ASSESSMENT_KEY)
        pre_scores = get_participant_scores(pre_df, PRE_ASSESSMENT_KEY)
        df_pre = pd.DataFrame(list(pre_results.items()), columns=['Question', 'Accuracy (%)'])
        df_pre['Phase'] = 'Pre-Webinar'
        df_pre['Question_Short'] = [f"Q{i+1}" for i in range(len(df_pre))]

    if post_file:
        post_df = pd.read_csv(post_file)
        post_results = grade_questions(post_df, POST_ASSESSMENT_KEY)
        post_scores = get_participant_scores(post_df, POST_ASSESSMENT_KEY)
        df_post = pd.DataFrame(list(post_results.items()), columns=['Question', 'Accuracy (%)'])
        df_post['Phase'] = 'Post-Webinar'
        df_post['Question_Short'] = [f"Q{i+1}" for i in range(len(df_post))]

    # --- Section 1: Executive Participant Overview ---
    st.subheader("Participant Overview")
    
    if pre_file and post_file:
        col_pre, col_post = st.columns(2)
        with col_pre:
            st.markdown("**Pre-Webinar Statistics**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Participants", len(pre_scores))
            c2.metric("Avg Score", f"{sum(pre_scores)/len(pre_scores):.1f} / 12")
            c3.metric("Highest Score", f"{max(pre_scores)} / 12")
        with col_post:
            st.markdown("**Post-Webinar Statistics**")
            c4, c5, c6 = st.columns(3)
            c4.metric("Total Participants", len(post_scores))
            
            pre_avg_score = sum(pre_scores)/len(pre_scores)
            post_avg_score = sum(post_scores)/len(post_scores)
            
            c5.metric("Avg Score", f"{post_avg_score:.1f} / 12", delta=f"{(post_avg_score - pre_avg_score):.1f}")
            c6.metric("Highest Score", f"{max(post_scores)} / 12", delta=f"{max(post_scores) - max(pre_scores)}")
            
    elif pre_file:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Participants", len(pre_scores))
        c2.metric("Avg Score", f"{sum(pre_scores)/len(pre_scores):.1f} / 12")
        c3.metric("Highest Score", f"{max(pre_scores)} / 12")
        
    elif post_file:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Participants", len(post_scores))
        c2.metric("Avg Score", f"{sum(post_scores)/len(post_scores):.1f} / 12")
        c3.metric("Highest Score", f"{max(post_scores)} / 12")

    st.markdown("---")

    # --- Section 2: Question Level Accuracy ---
    st.subheader("Question-Level Analysis")
    
    if pre_file and post_file:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=df_pre['Question_Short'], y=df_pre['Accuracy (%)'], 
                                  name='Pre-Webinar', marker_color='#1f77b4',
                                  marker_line_color='black', marker_line_width=1,  # Added Edge Color
                                  hovertext=df_pre['Question']))
        fig_comp.add_trace(go.Bar(x=df_post['Question_Short'], y=df_post['Accuracy (%)'], 
                                  name='Post-Webinar', marker_color='#2ca02c',
                                  marker_line_color='black', marker_line_width=1,  # Added Edge Color
                                  hovertext=df_post['Question']))
        fig_comp.update_layout(title="Pre vs Post Webinar Comparison per Question",
                               barmode='group', yaxis_range=[0, 100])
        st.plotly_chart(apply_grid(fig_comp), width="stretch")

    else:
        if pre_file:
            fig_pre = px.bar(df_pre.sort_values("Accuracy (%)"), x="Accuracy (%)", y="Question_Short", orientation='h',
                             title="Pre-Webinar - Performance Spread", hover_data=['Question'])
            fig_pre.update_traces(marker_line_color='black', marker_line_width=1)  # Added Edge Color
            fig_pre.update_xaxes(range=[0, 100])
            st.plotly_chart(apply_grid(fig_pre), width="stretch")
            
        if post_file:
            fig_post = px.bar(df_post.sort_values("Accuracy (%)"), x="Accuracy (%)", y="Question_Short", orientation='h',
                              title="Post-Webinar - Performance Spread", hover_data=['Question'])
            fig_post.update_traces(marker_line_color='black', marker_line_width=1)  # Added Edge Color
            fig_post.update_xaxes(range=[0, 100])
            st.plotly_chart(apply_grid(fig_post), width="stretch")
            
    # --- Section 3: Participant Score Distribution ---
    st.markdown("---")
    st.subheader("Participant Score Distribution (Frequency)")
    
    if pre_file and post_file:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=pre_scores, name='Pre-Webinar', marker_color='#1f77b4', opacity=0.75, 
                                        xbins=dict(start=-0.5, end=12.5, size=1), 
                                        marker_line_color='black', marker_line_width=1))  # Added Edge Color
        fig_dist.add_trace(go.Histogram(x=post_scores, name='Post-Webinar', marker_color='#2ca02c', opacity=0.75, 
                                        xbins=dict(start=-0.5, end=12.5, size=1), 
                                        marker_line_color='black', marker_line_width=1))  # Added Edge Color
        fig_dist.update_layout(barmode='overlay', title="Frequency of Participant Scores (Out of 12)", 
                               xaxis_title="Score (out of 12)", yaxis_title="Number of Teachers",
                               xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(apply_grid(fig_dist), width="stretch")
    else:
        if pre_file:
            fig_dist = px.histogram(x=pre_scores, title="Frequency of Pre-Webinar Scores",
                                    labels={'x': 'Score (out of 12)', 'y': 'Count'}, color_discrete_sequence=['#1f77b4'])
            fig_dist.update_traces(xbins=dict(start=-0.5, end=12.5, size=1), marker_line_color='black', marker_line_width=1)  # Added Edge Color
            fig_dist.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(apply_grid(fig_dist), width="stretch")
        if post_file:
            fig_dist = px.histogram(x=post_scores, title="Frequency of Post-Webinar Scores",
                                    labels={'x': 'Score (out of 12)', 'y': 'Count'}, color_discrete_sequence=['#2ca02c'])
            fig_dist.update_traces(xbins=dict(start=-0.5, end=12.5, size=1), marker_line_color='black', marker_line_width=1)  # Added Edge Color
            fig_dist.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
            st.plotly_chart(apply_grid(fig_dist), width="stretch")

    # --- Section 4: AI Insight Generation & Export ---
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
            
            # PDF Download Button
            pdf_bytes = create_pdf(st.session_state["generated_insights"])
            st.download_button(
                label="Download Insights Report (PDF)",
                data=pdf_bytes,
                file_name="Webinar_Insights_Report.pdf",
                mime="application/pdf",
                width="stretch"
            )