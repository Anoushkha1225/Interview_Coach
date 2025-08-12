# main.py - Enhanced FastAPI Backend with AI Rating System
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import PyPDF2
import docx
import re
import io
import statistics

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file.")

# Setup FastAPI app
app = FastAPI(title="Adaptive Interview Bot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    openai_api_key=api_key
)

# Global state (replace with DB in production)
sessions = {}

# ---------------- Models ----------------
class InterviewSetup(BaseModel):
    position: str
    experience_level: str
    session_id: str

class AnswerSubmission(BaseModel):
    answer: str
    question: str
    session_id: str

class SessionData(BaseModel):
    resume_chunks: List[str] = []
    resume_keywords: dict = {}
    resume_projects: List[str] = []
    question_queue: List[dict] = []
    asked_questions: List[dict] = []
    current_level: str = "Easy"
    position: str = ""
    experience_level: str = ""

# ---------------- Resume Processing ----------------
def read_pdf_from_bytes(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def read_docx_from_bytes(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

def chunk_resume(text, chunk_size=500):
    sentences = re.split(r'[.!?]\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def extract_keywords(text):
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb', 'aws',
        'azure', 'docker', 'kubernetes', 'git', 'machine learning', 'ai', 'data science', 'api'
    ]
    text_lower = text.lower()
    found_keywords = {k: text_lower.count(k) for k in tech_keywords if k in text_lower}
    companies = re.findall(r'(?:worked at|employed at|company:|@)\s*([A-Z][a-zA-Z\s&]+?)(?:\s*\n|\s*,|\s*\.|$)', text)
    experience_years = re.findall(r'(\d+)[\s\+]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', text_lower)
    return {'technical_skills': found_keywords, 'companies': [c.strip() for c in companies], 'experience_years': experience_years}

def extract_projects(text):
    return [p.strip() for p in re.findall(r'(?:project:|worked on)\s*([A-Z][a-zA-Z0-9\s&-]+)', text)]

# ---------------- AI Rating & Probability ----------------
async def generate_answer_review_and_rating(question, answer, position, experience_level):
    try:
        prompt = f"""
        You are a STRICT technical interviewer for "{position}" at {experience_level} level.
        Question: "{question}"
        Answer: "{answer}"
        Be HARSH. Rate 1-5, then give a 2-3 sentence critical review.
        Format:
        RATING: [1-5]
        REVIEW: [review]
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        if response and response.content:
            content = response.content.strip()
            rating_match = re.search(r'RATING:\s*(\d)', content)
            rating = int(rating_match.group(1)) if rating_match else 2
            review_match = re.search(r'REVIEW:\s*(.*)', content, re.DOTALL)
            review = review_match.group(1).strip() if review_match else "Needs significant improvement."
            return {"rating": rating, "review": review}
        return {"rating": 2, "review": "Answer requires improvement."}
    except Exception as e:
        print(f"Error in review: {e}")
        return {"rating": 2, "review": "Error generating review."}

async def calculate_hiring_probability(asked_questions, position, experience_level, total_questions):
    if not asked_questions:
        return {"probability": 0, "feedback": "No questions answered.", "average_rating": 0}
    completion_rate = len(asked_questions) / max(total_questions, 1)
    ratings = [q.get("rating", 1) for q in asked_questions]
    avg_rating = statistics.mean(ratings)
    if avg_rating >= 4.5: base_prob = 85
    elif avg_rating >= 4.0: base_prob = 70
    elif avg_rating >= 3.5: base_prob = 50
    elif avg_rating >= 3.0: base_prob = 30
    elif avg_rating >= 2.5: base_prob = 15
    elif avg_rating >= 2.0: base_prob = 8
    else: base_prob = 3
    completion_mult = min(1.0, completion_rate + 0.3)
    if completion_rate < 0.5: completion_mult *= 0.6
    elif completion_rate < 0.7: completion_mult *= 0.8
    final_prob = int(base_prob * completion_mult)
    return {
        "probability": final_prob,
        "feedback": f"Avg rating {avg_rating:.1f}, completion {completion_rate:.0%}",
        "average_rating": round(avg_rating, 1),
        "completion_rate": completion_rate
    }

# ---------------- Endpoints ----------------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), session_id: str = "default"):
    file_bytes = await file.read()
    if file.filename.endswith(".pdf"):
        text = read_pdf_from_bytes(file_bytes)
    elif file.filename.endswith(".docx"):
        text = read_docx_from_bytes(file_bytes)
    elif file.filename.endswith(".txt"):
        text = file_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text")
    if session_id not in sessions:
        sessions[session_id] = SessionData()
    s = sessions[session_id]
    s.resume_chunks = chunk_resume(text)
    s.resume_keywords = extract_keywords(text)
    s.resume_projects = extract_projects(text)
    return {"success": True, "chunks_count": len(s.resume_chunks), "projects_count": len(s.resume_projects), "keywords": s.resume_keywords}

@app.post("/setup-interview")
async def setup_interview(data: InterviewSetup):
    if data.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Upload resume first")
    s = sessions[data.session_id]
    s.position = data.position
    s.experience_level = data.experience_level
    s.question_queue = []
    for level in ["Easy", "Medium", "Hard"]:
        qs = await generate_questions(level, data.position, data.experience_level, s.resume_chunks)
        for q in qs:
            s.question_queue.append({"level": level, "question": q})
    return {"success": True, "total_questions": len(s.question_queue), "first_question": s.question_queue[0] if s.question_queue else None}

async def generate_questions(level, position, exp_level, resume_chunks):
    prompt = f"""
    Generate 6 {level} interview questions for '{position}' ({exp_level} level) based on this resume:
    {''.join(resume_chunks)}
    Numbered list only.
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return re.findall(r'\d+\.\s+(.*)', resp.content) if resp and resp.content else []

@app.get("/get-next-question/{session_id}")
async def get_next_question(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    asked = {q["question"] for q in s.asked_questions}
    for qd in s.question_queue:
        if qd["question"] not in asked:
            return {"success": True, "question": qd["question"], "level": qd["level"], "questions_answered": len(s.asked_questions), "total_questions": len(s.question_queue)}
    return {"success": False, "message": "No more questions", "interview_complete": True}

@app.post("/submit-answer")
async def submit_answer(data: AnswerSubmission):
    if data.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[data.session_id]
    relevance = await assess_answer_relevance(data.answer, data.question)
    review = await generate_answer_review_and_rating(data.question, data.answer, s.position, s.experience_level)
    s.current_level = adjust_question_difficulty(relevance, s.current_level)
    s.asked_questions.append({"question": data.question, "answer": data.answer, "level": s.current_level, "relevance": relevance, "rating": review["rating"], "review": review["review"]})
    return {"success": True, "relevance": relevance, **review, "questions_answered": len(s.asked_questions), "total_questions": len(s.question_queue)}

async def assess_answer_relevance(answer, question):
    prompt = f"Is this answer relevant to the question? Q: {question} A: {answer}. Reply only 'relevant' or 'not_relevant'."
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip().lower() if resp and resp.content else "relevant"

def adjust_question_difficulty(relevance, current_level):
    levels = ["Easy", "Medium", "Hard"]
    idx = levels.index(current_level)
    if relevance == "relevant" and idx < 2: return levels[idx+1]
    if relevance == "not_relevant" and idx > 0: return levels[idx-1]
    return current_level

@app.get("/interview-summary/{session_id}")
async def interview_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    hiring = await calculate_hiring_probability(s.asked_questions, s.position, s.experience_level, len(s.question_queue))
    return {"success": True, "position": s.position, "experience_level": s.experience_level, "asked_questions": s.asked_questions, "overall_rating": hiring["average_rating"], "hiring_probability": hiring["probability"], "hiring_feedback": hiring["feedback"], "completion_rate": hiring["completion_rate"]}

@app.delete("/reset-session/{session_id}")
async def reset_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"success": True}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
