import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import streamlit as st
import uuid
import requests
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import re
import hashlib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Personalized Cover Letter Generator",
    page_icon="💜",
    layout="wide"
)

# ---------------- SESSION STATE INIT ----------------
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

if "extra_instructions" not in st.session_state:
    st.session_state.extra_instructions = ""

# ---------------- LAZY OCR LOADING ----------------
def get_ocr():
    if "ocr_model" not in st.session_state:
        with st.spinner("⚙️ Initializing OCR engine (one-time setup)..."):
            st.session_state.ocr_model = PaddleOCR(lang='en')
    return st.session_state.ocr_model

# ---------------- NAME EXTRACTION ----------------
def extract_candidate_name(resume_text):
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    for line in lines[:10]:
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+){1,3}$", line):
            return line
    return "Applicant"

# ---------------- OCR FUNCTIONS ----------------
def paddle_ocr_from_image(pil_image):
    ocr_model = get_ocr()
    img = np.array(pil_image)
    result = ocr_model.ocr(img, cls=True)
    text = ""
    for line in result:
        for word_info in line:
            text += word_info[1][0] + " "
        text += "\n"
    return text

def extract_text_from_pdf(file):
    file.seek(0)
    text = ""
    reader = PdfReader(file)

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if not text.strip():
        file.seek(0)
        images = convert_from_bytes(file.read())
        for img in images:
            text += paddle_ocr_from_image(img) + "\n"

    return text

def extract_text_from_image(file):
    file.seek(0)
    image = Image.open(file).convert("RGB")
    return paddle_ocr_from_image(image)

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type.startswith("image/"):
        return extract_text_from_image(file)
    elif file.type == "text/plain":
        file.seek(0)
        return file.read().decode("utf-8")
    return ""

# ---------------- FIXED LLAMA CALL ----------------
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
        response = requests.post(url, json=payload, timeout=120)

        if response.status_code != 200:
            return f"❌ Ollama API Error {response.status_code}: {response.text}"

        data = response.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return f"❌ Unexpected API response format: {data}"

    except requests.exceptions.ConnectionError:
        return "❌ Connection Error: Ollama server not reachable at http://localhost:11434"

    except requests.exceptions.Timeout:
        return "❌ Timeout Error: Ollama took too long to respond"

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"

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
    st.session_state.extra_instructions = ""

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
        if st.button(chat["title"], key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id

# ---------------- MAIN UI ----------------
current_chat = get_current_chat()
st.title("🎯 Personalized Cover Letter Generator")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader(
        "📄 Upload Resume",
        type=["pdf", "jpg", "jpeg", "png", "txt"],
        key=f"resume_uploader_{st.session_state.current_chat_id}"
    )

with col2:
    jd_file = st.file_uploader(
        "🧾 Upload Job Description",
        type=["pdf", "jpg", "jpeg", "png", "txt"],
        key=f"jd_uploader_{st.session_state.current_chat_id}"
    )

# -------- OPTIONAL TEXTBOX --------
st.session_state.extra_instructions = st.text_area(
    "✏️ Additional Instructions (Optional)",
    placeholder="E.g., emphasize AI/ML projects, keep it concise, highlight leadership..."
)

# ---------------- PROCESS RESUME ----------------
if resume_file:
    file_bytes = resume_file.getvalue()
    file_id = hashlib.md5(file_bytes).hexdigest()
    key = ("resume", st.session_state.current_chat_id, file_id)

    if key not in st.session_state.processed_files:
        with st.spinner("🔍 Processing resume..."):
            resume_file.seek(0)
            text = extract_text(resume_file)

        st.session_state.resume_text = text
        st.session_state.candidate_name = extract_candidate_name(text)

        current_chat["messages"].append(
            ("assistant", f"✅ Resume processed: {resume_file.name}")
        )

        st.session_state.processed_files.add(key)

# ---------------- PROCESS JD ----------------
if jd_file:
    file_bytes = jd_file.getvalue()
    file_id = hashlib.md5(file_bytes).hexdigest()
    key = ("jd", st.session_state.current_chat_id, file_id)

    if key not in st.session_state.processed_files:
        with st.spinner("📑 Processing job description..."):
            jd_file.seek(0)
            jd_text = extract_text(jd_file)

        st.session_state.job_description_text = jd_text

        current_chat["messages"].append(
            ("assistant", f"✅ Job description processed: {jd_file.name}")
        )

        st.session_state.processed_files.add(key)

# ---------------- DISPLAY CHAT ----------------
for role, msg in current_chat["messages"]:
    with st.chat_message(role):
        st.write(msg)

# ---------------- GENERATE COVER LETTER ----------------
if st.button("Generate Cover Letter"):
    if not st.session_state.resume_text:
        current_chat["messages"].append(
            ("assistant", "⚠️ Please upload a resume file.")
        )
    elif not st.session_state.job_description_text:
        current_chat["messages"].append(
            ("assistant", "⚠️ Please upload a job description file.")
        )
    else:
        extra_section = ""
        if st.session_state.extra_instructions.strip():
            extra_section = f"\nAdditional Instructions:\n{st.session_state.extra_instructions}\n"

        system_prompt = f"""
You are a professional career assistant.

Write a highly personalized, ATS-optimized cover letter tailored to the job description.

Candidate Name: {st.session_state.candidate_name}

Resume:
{st.session_state.resume_text}

Job Description:
{st.session_state.job_description_text}
{extra_section}
Strict Rules:
- Do NOT invent experience
- Align resume skills with job requirements
- Use keywords from job description naturally
- Maintain formal business letter format
- No placeholders
- End with:
  Sincerely,
  {st.session_state.candidate_name}
"""

        with st.spinner("✍️ Generating tailored cover letter..."):
            reply = call_llama(system_prompt, "Generate the cover letter.")

        current_chat["messages"].append(("assistant", reply))
        st.rerun()