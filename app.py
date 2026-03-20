import streamlit as st
import pandas as pd
import PyPDF2
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. AI KNOWLEDGE BASE (REAL INDUSTRY DATA) ---
JOB_ROLES = {
    "Frontend Developer": {
        "skills": ["html", "css", "javascript", "react", "vue", "typescript", "figma", "sass", "nextjs", "tailwind"],
        "desc": "Focuses on user-facing interface and design implementation.",
        "courses": ["Meta Front-End Developer Certificate (Coursera)", "FreeCodeCamp Responsive Web Design", "Frontend Masters - React Path"],
        "certs": ["Meta Front-End Certificate", "Google UX Design Certificate"]
    },
    "Backend Developer": {
        "skills": ["python", "nodejs", "sql", "api", "express", "django", "docker", "aws", "mongodb", "redis", "microservices"],
        "desc": "Handles server-side logic, databases, and API architecture.",
        "courses": ["Node.js Developer Roadmap (roadmap.sh)", "Python for Everybody (UMich)", "Back-End Engineer Path (Codecademy)"],
        "certs": ["AWS Certified Developer", "Oracle Certified Professional: Java"]
    },
    "Fullstack Developer": {
        "skills": ["react", "nodejs", "sql", "javascript", "html", "css", "docker", "aws", "typescript", "git", "api"],
        "desc": "Manages both client-side and server-side development.",
        "courses": ["Full Stack Open (University of Helsinki)", "The Web Developer Bootcamp (Udemy)"],
        "certs": ["AWS Certified Solutions Architect", "IBM Full Stack Developer"]
    },
    "Data Scientist": {
        "skills": ["python", "machine learning", "statistics", "sql", "pandas", "deep learning", "tensorflow", "pytorch", "tableau"],
        "desc": "Uses advanced analytics and AI to interpret complex data.",
        "courses": ["Machine Learning Specialization (Andrew Ng)", "Kaggle Micro-courses", "Data Science MicroMasters (edX)"],
        "certs": ["Google Data Analytics Professional", "IBM Data Science Professional"]
    },
    "Civil Engineer": {
        "skills": ["autocad", "staad pro", "surveying", "concrete", "structural analysis", "revit", "estimation", "construction"],
        "desc": "Designs and manages physical infrastructure projects.",
        "courses": ["Autodesk Revit Architecture (LinkedIn)", "Staad.Pro Structural Analysis (Udemy)"],
        "certs": ["EIT (Engineer In Training)", "PMP (Project Management)"]
    },
    "UI/UX Designer": {
        "skills": ["figma", "adobe xd", "wireframing", "prototyping", "user research", "photoshop", "illustrator"],
        "desc": "Designs intuitive and aesthetic digital user journeys.",
        "courses": ["Google UX Design Professional Certificate", "Interaction Design Foundation"],
        "certs": ["Adobe Certified Professional", "Google UX Design Certificate"]
    },
    "DevOps Engineer": {
        "skills": ["docker", "kubernetes", "linux", "jenkins", "aws", "terraform", "ansible", "ci/cd", "bash"],
        "desc": "Specializes in automation and deployment infrastructure.",
        "courses": ["Cloud Dev Ops Nanodegree (Udacity)", "Docker and Kubernetes Guide"],
        "certs": ["CKA (Certified Kubernetes Admin)", "AWS DevOps Engineer Pro"]
    },
    "Mechanical Engineer": {
        "skills": ["solidworks", "cad", "thermodynamics", "manufacturing", "mechanical design", "ansys", "matlab", "robotics"],
        "desc": "Designs, develops, and tests mechanical devices.",
        "courses": ["SolidWorks Mastery (Udemy)", "Mechanical Engineering Design (MIT)"],
        "certs": ["CSWP (Certified SOLIDWORKS Pro)", "PE License"]
    },
    "Software Tester": {
        "skills": ["selenium", "manual testing", "automation", "unit testing", "bug tracking", "quality assurance", "cypress"],
        "desc": "Ensures software quality through rigorous testing.",
        "courses": ["Selenium WebDriver with Java", "Complete Guide to Manual Testing"],
        "certs": ["ISTQB Certified Tester", "LambdaTest Selenium Cert"]
    }
}

# --- 2. CORE AI ENGINE ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def get_role_similarity(resume_text):
    """Real AI Logic: Compare resume against all roles using keyword density"""
    results = []
    for role, data in JOB_ROLES.items():
        match_count = sum(1 for skill in data['skills'] if skill in resume_text)
        percentage = (match_count / len(data['skills'])) * 100
        results.append({"role": role, "score": percentage, "desc": data['desc'], "skills": data['skills']})
    return sorted(results, key=lambda x: x['score'], reverse=True)

# --- 3. PROFESSIONAL STYLING (CSS) ---
st.set_page_config(page_title="Pro AI Resume Analyzer", layout="wide")
st.markdown("""
    <style>
    .card {
        background-color: #ffffff; padding: 25px; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 25px;
        color: #1e1e1e !important;
    }
    .card h3, .card h4, .card p, .card b { color: #1e1e1e !important; }
    .match-score-container {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; padding: 40px; border-radius: 15px; text-align: center; margin-bottom: 30px;
    }
    .badge {
        padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; 
        margin: 4px; display: inline-block; border: 1px solid rgba(0,0,0,0.1);
    }
    .badge-relevant { background-color: #dcfce7; color: #166534; }
    .badge-missing { background-color: #fee2e2; color: #991b1b; }
    .badge-neutral { background-color: #f3f4f6; color: #374151; }
    .warning-box {
        background-color: #fffbeb; border-left: 6px solid #f59e0b; 
        padding: 20px; border-radius: 8px; color: #92400e; margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN APP UI ---
st.title("💼 Pro AI Resume Analyzer")
st.caption("Building your personal data-driven career path.")

col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    uploaded_file = st.file_uploader("📤 Upload Resume (PDF format)", type="pdf")
with col_input2:
    target_role = st.selectbox("🎯 Select Applied Role", ["-- Select a Role --"] + list(JOB_ROLES.keys()))

if uploaded_file and target_role != "-- Select a Role --":
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # 1. RUN AI ANALYSIS
    all_role_matches = get_role_similarity(resume_text)
    current_role_data = next(item for item in all_role_matches if item["role"] == target_role)
    detected_role_data = all_role_matches[0] 
    
    # UI: MATCH SCORE
    st.markdown(f"""
        <div class="match-score-container">
            <h1 style="color:white; margin:0;">{round(current_role_data['score'])}% Compatibility</h1>
            <p style="color:white; opacity:0.9;">Analyzing qualifications for {target_role}</p>
        </div>
    """, unsafe_allow_html=True)

    # UI: ROLE MISMATCH WARNING
    if detected_role_data['role'] != target_role:
        st.markdown(f"""
            <div class="warning-box">
                <b>⚠️ Role Mismatch Detected</b><br>
                This resume is mathematically more aligned with <b>{detected_role_data['role']}</b>. 
                Applying for <b>{target_role}</b> may be difficult with your current skill set.
            </div>
        """, unsafe_allow_html=True)

    # UI: DETAILED ANALYSIS (SIDE-BY-SIDE BADGES)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f'<div class="card"><h3>📋 Required Skills: {target_role}</h3>', unsafe_allow_html=True)
        req_badges = "".join([f'<span class="badge {"badge-relevant" if s in resume_text else "badge-neutral"}">{s.upper()}</span>' for s in JOB_ROLES[target_role]['skills']])
        st.markdown(f'<div style="display:flex; flex-wrap:wrap;">{req_badges}</div></div>', unsafe_allow_html=True)

    with c2:
        found = [s for s in JOB_ROLES[target_role]['skills'] if s in resume_text]
        missing = [s for s in JOB_ROLES[target_role]['skills'] if s not in resume_text]
        
        st.markdown('<div class="card"><h3>📊 Skills Gap Analysis</h3>', unsafe_allow_html=True)
        st.markdown(f"<b>✅ Found ({len(found)})</b>", unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex; flex-wrap:wrap; margin-bottom:10px;">' + "".join([f'<span class="badge badge-relevant">{s.upper()}</span>' for s in found]) + '</div>', unsafe_allow_html=True)
        
        st.markdown(f"<b>➕ Missing to Upgrade ({len(missing)})</b>", unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex; flex-wrap:wrap;">' + "".join([f'<span class="badge badge-missing">+ {s.upper()}</span>' for s in missing]) + '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # UI: AI RECOMMENDATIONS (NON-MOCKED)
    st.subheader("💡 Best Suited Alternative Roles")
    rec_cols = st.columns(3)
    recommendations = [r for r in all_role_matches if r['role'] != target_role][:3]
    
    for idx, rec in enumerate(recommendations):
        with rec_cols[idx]:
            st.markdown(f"""<div class="card"><b>{rec['role']}</b><br><span style="color:#166534; font-size:0.8em;">{round(rec['score'])}% Match</span><p style="font-size:0.8em; margin-top:5px;">{rec['desc']}</p></div>""", unsafe_allow_html=True)

    # UI: CAREER GROWTH ROADMAP (SUGGESTIONS)
    st.divider()
    st.markdown("## 📖 Career Growth Roadmap")
    s_col1, s_col2 = st.columns([2, 1])
    
    with s_col1:
        st.markdown(f'<div class="card"><h4>🚀 Priority Learning Plan</h4>', unsafe_allow_html=True)
        if missing:
            for m in missing[:3]: st.write(f"🔹 **Master {m.upper()}:** Essential for {target_role} roles.")
            st.markdown("<br><b>Top Courses:</b>", unsafe_allow_html=True)
            for course in JOB_ROLES[target_role]['courses']: st.write(f"📖 {course}")
        else: st.success("You are fully qualified for this role!")
        st.markdown('</div>', unsafe_allow_html=True)

    with s_col2:
        st.markdown(f'<div class="card"><h4>🏅 Certifications</h4>', unsafe_allow_html=True)
        for cert in JOB_ROLES[target_role]['certs']: st.info(f"✔️ {cert}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 5. SAVE TO LOCAL DATASET ---
    df_db = pd.DataFrame([[target_role, resume_text]], columns=['role', 'resume_text'])
    df_db.to_csv("user_resume_dataset.csv", mode='a', header=not os.path.exists("user_resume_dataset.csv"), index=False)

else:
    st.info("👋 Welcome! Please upload your PDF resume and select a target role in the dropdown to begin analysis.")