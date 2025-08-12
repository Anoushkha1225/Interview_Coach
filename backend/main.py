# main.py - Enhanced FastAPI Backend with AI Rating System (FIXED)
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import PyPDF2
import docx
import re
import io
import statistics
import traceback

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file.")

# Setup FastAPI app
app = FastAPI(title="Adaptive Interview Bot API", version="1.0.0")

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://interview-coach-chi.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Setup LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    openai_api_key=api_key
)

# Global state (in production, use a database)
sessions = {}

# Pydantic models
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

# Resume processing functions
def read_pdf_from_bytes(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text.strip() else None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        traceback.print_exc()
        return None

def read_docx_from_bytes(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
        return text if text.strip() else None
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        traceback.print_exc()
        return None

def chunk_resume(text, chunk_size=500):
    if not text or not text.strip():
        return []
    
    sentences = re.split(r'[.!?]\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
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
    if not text:
        return {'technical_skills': {}, 'companies': [], 'experience_years': []}
    
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb', 'aws',
        'azure', 'docker', 'kubernetes', 'git', 'machine learning', 'ai', 'data science', 'api'
    ]
    text_lower = text.lower()
    found_keywords = {k: text_lower.count(k) for k in tech_keywords if k in text_lower}
    companies = re.findall(r'(?:worked at|employed at|company:|@)\s*([A-Z][a-zA-Z\s&]+?)(?:\s*\n|\s*,|\s*\.|$)', text)
    experience_years = re.findall(r'(\d+)[\s\+]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', text_lower)
    
    return {
        'technical_skills': found_keywords, 
        'companies': [c.strip() for c in companies], 
        'experience_years': experience_years
    }

def extract_projects(text):
    if not text:
        return []
    projects = re.findall(r'(?:project:|worked on)\s*([A-Z][a-zA-Z0-9\s&-]+)', text)
    return [p.strip() for p in projects]

# AI Rating Functions
async def generate_answer_review_and_rating(question, answer, position, experience_level):
    try:
        review_prompt = f"""
        You are a STRICT technical interviewer for the position of "{position}" at {experience_level} level.
        
        Question asked: "{question}"
        Candidate's answer: "{answer}"
        
        Be VERY STRICT in your evaluation. This is a real technical interview scenario.
        
        Rating criteria (BE HARSH):
        - 5 stars: EXCEPTIONAL - Perfect answer with deep insights, covers all aspects, shows mastery level understanding
        - 4 stars: GOOD - Solid answer covering most important points with good depth and accuracy  
        - 3 stars: ADEQUATE - Basic correct answer but lacks depth, missing some key points
        - 2 stars: POOR - Partially correct but significant gaps, shallow understanding, major missing elements
        - 1 star: VERY POOR - Wrong, off-topic, demonstrates lack of understanding, or extremely incomplete
        
        Format your response as:
        RATING: [1-5]
        REVIEW: [Your critical review focusing on gaps and improvements needed]
        """
        
        response = llm.invoke([HumanMessage(content=review_prompt)])
        if response and response.content:
            content = response.content.strip()
            
            # Extract rating
            rating_match = re.search(r'RATING:\s*(\d)', content)
            rating = int(rating_match.group(1)) if rating_match else 2
            
            # Extract review
            review_match = re.search(r'REVIEW:\s*(.*)', content, re.DOTALL)
            review = review_match.group(1).strip() if review_match else "Answer needs significant improvement to meet industry standards."
            
            return {"rating": rating, "review": review}
        
        return {"rating": 2, "review": "Answer requires substantial improvement to meet professional interview standards."}
        
    except Exception as e:
        print(f"Error generating review and rating: {e}")
        return {"rating": 2, "review": "Unable to generate detailed review. Answer needs significant improvement."}

async def calculate_hiring_probability(asked_questions, position, experience_level, total_questions_available):
    try:
        if not asked_questions:
            return {"probability": 0, "feedback": "No questions answered. Interview incomplete.", "average_rating": 0}
        
        completion_rate = len(asked_questions) / max(total_questions_available, 1)
        
        if completion_rate < 0.3:
            return {
                "probability": min(5, int(completion_rate * 100 * 0.2)),
                "feedback": f"Interview significantly incomplete ({len(asked_questions)}/{total_questions_available} questions answered). Insufficient data to make a hiring decision.",
                "average_rating": 0,
                "completion_rate": completion_rate
            }
        
        ratings = [q.get('rating', 1) for q in asked_questions]
        avg_rating = statistics.mean(ratings)
        
        # Calculate base probability from performance
        if avg_rating >= 4.5:
            base_probability = 85
        elif avg_rating >= 4.0:
            base_probability = 70
        elif avg_rating >= 3.5:
            base_probability = 50
        elif avg_rating >= 3.0:
            base_probability = 30
        elif avg_rating >= 2.5:
            base_probability = 15
        elif avg_rating >= 2.0:
            base_probability = 8
        else:
            base_probability = 3
        
        # Apply completion rate multiplier
        completion_multiplier = min(1.0, completion_rate + 0.3)
        if completion_rate < 0.5:
            completion_multiplier *= 0.6
        elif completion_rate < 0.7:
            completion_multiplier *= 0.8
        
        final_probability = int(base_probability * completion_multiplier)
        final_probability = max(0, min(100, final_probability))
        
        feedback = f"Average rating: {avg_rating:.1f}/5.0, Completion: {completion_rate:.1%}"
        if completion_rate < 0.5:
            feedback = f"Interview only {completion_rate:.1%} complete ({len(asked_questions)}/{total_questions_available} questions), which significantly impacts hiring potential. " + feedback
        
        return {
            "probability": final_probability, 
            "feedback": feedback, 
            "average_rating": round(avg_rating, 1),
            "completion_rate": completion_rate,
            "questions_answered": len(asked_questions),
            "total_questions": total_questions_available
        }
        
    except Exception as e:
        print(f"Error calculating hiring probability: {e}")
        return {
            "probability": 0, 
            "feedback": "Unable to calculate hiring probability due to technical error.", 
            "average_rating": 0,
            "completion_rate": 0
        }

# API endpoints
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), session_id: Optional[str] = Form("default")):
    try:
        print(f"Received file upload request - Filename: {file.filename}, Content-Type: {file.content_type}, Session ID: {session_id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (10MB limit)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        print(f"File size: {len(file_content)} bytes")
        
        # Process based on file type
        text = None
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith('.pdf'):
            print("Processing PDF file...")
            text = read_pdf_from_bytes(file_content)
        elif filename_lower.endswith('.docx'):
            print("Processing DOCX file...")
            text = read_docx_from_bytes(file_content)
        elif filename_lower.endswith('.txt'):
            print("Processing TXT file...")
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = file_content.decode('latin-1')
                except UnicodeDecodeError:
                    raise HTTPException(status_code=400, detail="Unable to decode text file. Please use UTF-8 encoding.")
        else:
            supported_formats = ['.pdf', '.docx', '.txt']
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            )
        
        print(f"Extracted text length: {len(text) if text else 0}")
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file or file is empty")
        
        # Process resume
        resume_chunks = chunk_resume(text)
        resume_keywords = extract_keywords(text)
        resume_projects = extract_projects(text)
        
        print(f"Processing results - Chunks: {len(resume_chunks)}, Projects: {len(resume_projects)}")
        
        # Store in session
        if session_id not in sessions:
            sessions[session_id] = SessionData()
        
        sessions[session_id].resume_chunks = resume_chunks
        sessions[session_id].resume_keywords = resume_keywords
        sessions[session_id].resume_projects = resume_projects
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Resume processed successfully. Found {len(resume_chunks)} chunks and {len(resume_projects)} projects.",
                "chunks_count": len(resume_chunks),
                "projects_count": len(resume_projects),
                "keywords": resume_keywords
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error processing resume: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/setup-interview")
async def setup_interview(setup: InterviewSetup):
    try:
        session_id = setup.session_id
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please upload resume first.")
        
        session = sessions[session_id]
        session.position = setup.position
        session.experience_level = setup.experience_level
        
        # Generate questions for all difficulty levels
        question_queue = []
        
        for category in ['Easy', 'Medium', 'Hard']:
            questions = await generate_questions(category, setup.position, setup.experience_level, session.resume_chunks)
            for q in questions:
                question_queue.append({"level": category, "question": q})
        
        session.question_queue = question_queue
        
        return {
            "success": True,
            "message": "Interview setup complete",
            "total_questions": len(question_queue),
            "first_question": question_queue[0] if question_queue else None
        }
    
    except Exception as e:
        print(f"Error setting up interview: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error setting up interview: {str(e)}")

async def generate_questions(category, position, experience_level, resume_chunks):
    try:
        prompt = f"""
        You are an AI Interview Question Generator.
        Based on the following resume chunks, generate 6 {category} interview questions for the role of '{position}' suitable for a(n) {experience_level} level candidate.
        
        Resume Content:
        {''.join(resume_chunks)}
        
        Only return the questions in a numbered list format.
        If it is an IT/software related role, Make sure to ask technical questions, they should be based on the level of experience.
        If it is a non-IT role, focus on behavioral and situational questions.
        Format:
        1. Question one
        2. Question two 
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        if response and response.content:
            questions = re.findall(r'\d+\.\s+(.*)', response.content)
            return questions
        return []
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

@app.get("/get-next-question/{session_id}")
async def get_next_question(session_id: str):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        # Find next question that hasn't been asked
        asked_question_texts = {q["question"] for q in session.asked_questions}
        
        for question_data in session.question_queue:
            if question_data["question"] not in asked_question_texts:
                return {
                    "success": True,
                    "question": question_data["question"],
                    "level": question_data["level"],
                    "current_level": session.current_level,
                    "questions_answered": len(session.asked_questions),
                    "total_questions": len(session.question_queue)
                }
        
        return {
            "success": False,
            "message": "No more questions available",
            "interview_complete": True
        }
    
    except Exception as e:
        print(f"Error getting next question: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting next question: {str(e)}")

@app.post("/submit-answer")
async def submit_answer(submission: AnswerSubmission):
    try:
        session_id = submission.session_id
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        # Assess answer relevance
        relevance = await assess_answer_relevance(submission.answer, submission.question)
        
        # Generate AI review and rating
        review_data = await generate_answer_review_and_rating(
            submission.question, 
            submission.answer, 
            session.position, 
            session.experience_level
        )
        
        # Adjust difficulty level
        new_level = adjust_question_difficulty(relevance, session.current_level)
        session.current_level = new_level
        
        # Store the asked question and answer with review and rating
        session.asked_questions.append({
            "question": submission.question,
            "answer": submission.answer,
            "level": session.current_level,
            "relevance": relevance,
            "rating": review_data["rating"],
            "review": review_data["review"]
        })
        
        return {
            "success": True,
            "relevance": relevance,
            "new_level": new_level,
            "rating": review_data["rating"],
            "review": review_data["review"],
            "questions_answered": len(session.asked_questions),
            "total_questions": len(session.question_queue)
        }
    
    except Exception as e:
        print(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting answer: {str(e)}")

async def assess_answer_relevance(answer, question):
    try:
        relevance_prompt = f"""
        The candidate answered: '{answer}'
        Determine if this answer is 'relevant' or 'not_relevant' based on the context of this question: '{question}'.
        Only return 'relevant' or 'not_relevant'.
        """
        
        relevance_response = llm.invoke([HumanMessage(content=relevance_prompt)])
        if relevance_response and relevance_response.content:
            return relevance_response.content.strip().lower()
        return 'relevant'
    except Exception as e:
        print(f"Error assessing relevance: {e}")
        return 'relevant'

def adjust_question_difficulty(last_answer_relevance, current_level):
    levels = ['Easy', 'Medium', 'Hard']
    idx = levels.index(current_level)
    
    if last_answer_relevance == 'relevant' and idx < len(levels) - 1:
        return levels[idx + 1]
    elif last_answer_relevance == 'not_relevant' and idx > 0:
        return levels[idx - 1]
    return current_level

@app.get("/interview-summary/{session_id}")
async def get_interview_summary(session_id: str):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        # Calculate hiring probability
        hiring_assessment = await calculate_hiring_probability(
            session.asked_questions, 
            session.position, 
            session.experience_level,
            len(session.question_queue)
        )
        
        return {
            "success": True,
            "position": session.position,
            "experience_level": session.experience_level,
            "questions_answered": len(session.asked_questions),
            "total_questions": len(session.question_queue),
            "asked_questions": session.asked_questions,
            "final_level": session.current_level,
            "overall_rating": hiring_assessment.get("average_rating", 0),
            "hiring_probability": hiring_assessment.get("probability", 0),
            "hiring_feedback": hiring_assessment.get("feedback", ""),
            "completion_rate": hiring_assessment.get("completion_rate", 0)
        }
    
    except Exception as e:
        print(f"Error getting interview summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting interview summary: {str(e)}")

@app.delete("/reset-session/{session_id}")
async def reset_session(session_id: str):
    try:
        if session_id in sessions:
            del sessions[session_id]
        
        return {"success": True, "message": "Session reset successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting session: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Interview Bot API is running"}

# Add a test endpoint to verify the API is working
@app.get("/")
async def root():
    return {"message": "Interview Coach API is running", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)