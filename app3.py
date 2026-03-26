import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import streamlit as st
import requests
import uuid
import hashlib
import json

from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="💜",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "job_description_text" not in st.session_state:
    st.session_state.job_description_text = ""

if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = "Applicant"

# ---------------- CHAT MANAGEMENT ----------------
def start_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": []
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.resume_text = ""
    st.session_state.job_description_text = ""
    st.session_state.candidate_name = "Applicant"
    st.session_state.processed_files = set()

def get_current_chat():
    if st.session_state.current_chat_id is None:
        start_new_chat()
    return st.session_state.chats[st.session_state.current_chat_id]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("💬 Chats")

    if st.button("+ New Chat"):
        start_new_chat()

    st.divider()

    for chat_id, chat in st.session_state.chats.items():
        if st.button(chat["title"], key=chat_id):
            st.session_state.current_chat_id = chat_id

# ---------------- OCR ----------------
def get_ocr():
    if "ocr_model" not in st.session_state:
        st.session_state.ocr_model = PaddleOCR(lang='en')
    return st.session_state.ocr_model

def paddle_ocr_from_image(img):
    ocr = get_ocr()
    result = ocr.ocr(np.array(img), cls=True)
    text = ""
    for line in result:
        for word in line:
            text += word[1][0] + " "
        text += "\n"
    return text

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        images = convert_from_bytes(file.read())
        for img in images:
            text += paddle_ocr_from_image(img)

    return text

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type.startswith("image"):
        img = Image.open(file).convert("RGB")
        return paddle_ocr_from_image(img)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

# ---------------- NAME ----------------
def extract_candidate_name(text):
    for line in text.split("\n")[:10]:
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+){1,3}$", line.strip()):
            return line.strip()
    return "Applicant"

# ---------------- MATCH SCORE ----------------
def calculate_match_score(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    vecs = vectorizer.fit_transform([resume, jd])
    return round(cosine_similarity(vecs[0], vecs[1])[0][0] * 100, 2)

# ---------------- LLAMA ----------------
def call_llama(system_prompt, user_prompt):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    try:
        res = requests.post(url, json=payload, timeout=120)
        data = res.json()
        return data.get("message", {}).get("content", "❌ Error")
    except Exception as e:
        return f"❌ {str(e)}"

# ---------------- MAIN ----------------
current_chat = get_current_chat()

st.title("🎯 AI Career Assistant")

col1, col2 = st.columns(2)

with col1:
    resume = st.file_uploader("Upload Resume", key=f"res_{st.session_state.current_chat_id}")

with col2:
    jd = st.file_uploader("Upload Job Description", key=f"jd_{st.session_state.current_chat_id}")

style = st.selectbox(
    "Cover Letter Style",
    ["Professional", "Technical", "Concise", "Creative"]
)

extra = st.text_area("Extra Instructions (optional)")

# ---------------- PROCESS FILES ----------------
if resume:
    file_id = hashlib.md5(resume.getvalue()).hexdigest()
    key = ("resume", file_id)

    if key not in st.session_state.processed_files:
        text = extract_text(resume)
        st.session_state.resume_text = text
        st.session_state.candidate_name = extract_candidate_name(text)

        current_chat["messages"].append(("assistant", f"✅ Resume uploaded: {resume.name}"))
        st.session_state.processed_files.add(key)

if jd:
    file_id = hashlib.md5(jd.getvalue()).hexdigest()
    key = ("jd", file_id)

    if key not in st.session_state.processed_files:
        text = extract_text(jd)
        st.session_state.job_description_text = text

        current_chat["messages"].append(("assistant", f"✅ JD uploaded: {jd.name}"))
        st.session_state.processed_files.add(key)

# ---------------- DISPLAY CHAT ----------------
for role, msg in current_chat["messages"]:
    with st.chat_message(role):
        st.write(msg)

# ---------------- COVER LETTER ----------------
if st.button("Generate Cover Letter"):
    if not st.session_state.resume_text:
        current_chat["messages"].append(("assistant", "⚠️ Upload resume first"))

    elif not st.session_state.job_description_text:
        current_chat["messages"].append(("assistant", "⚠️ Upload job description first"))

    else:
        prompt = f"""
Write a personalized ATS-optimized cover letter.

Name: {st.session_state.candidate_name}
Style: {style}

Resume:
{st.session_state.resume_text}

Job Description:
{st.session_state.job_description_text}

Extra:
{extra}

End with:
Sincerely,
{st.session_state.candidate_name}
"""

        reply = call_llama("Career assistant", prompt)

        current_chat["messages"].append(("assistant", "📄 Cover Letter Generated:\n"))
        current_chat["messages"].append(("assistant", reply))

    st.rerun()

# ---------------- INTERVIEW ----------------
if st.button("Generate Interview Questions"):

    if not st.session_state.resume_text or not st.session_state.job_description_text:
        current_chat["messages"].append(("assistant", "⚠️ Upload both resume and JD"))

    else:
        prompt = f"""
Generate 10 interview questions with answers.

Resume:
{st.session_state.resume_text}

JD:
{st.session_state.job_description_text}
"""

        reply = call_llama("Expert interviewer", prompt)

        current_chat["messages"].append(("assistant", "🎤 Interview Questions:\n"))
        current_chat["messages"].append(("assistant", reply))

    st.rerun()

# ---------------- MATCH SCORE ----------------
if st.session_state.resume_text and st.session_state.job_description_text:
    score = calculate_match_score(
        st.session_state.resume_text,
        st.session_state.job_description_text
    )
    st.metric("Match Score", f"{score}%")