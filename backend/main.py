# app.py - Enhanced FastAPI Backend with AI Rating System
import os
import pyttsx3
from fastapi import FastAPI, File, UploadFile, HTTPException
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
from collections import deque
import io
import tempfile
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
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free",
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

# Resume processing functions (from your original code)
def read_pdf_from_bytes(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def read_docx_from_bytes(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

def chunk_resume(text, chunk_size=500):
    sentences = re.split(r'[.!?]\s+', text)
    chunks = []
    current_chunk = ""
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
    tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'sql', 'mongodb', 'aws', 'azure', 
                     'docker', 'kubernetes', 'git', 'machine learning', 'ai', 'data science', 'api']
    text_lower = text.lower()
    found_keywords = {k: text_lower.count(k) for k in tech_keywords if k in text_lower}
    companies = re.findall(r'(?:worked at|employed at|company:|@)\s*([A-Z][a-zA-Z\s&]+?)(?:\s*\n|\s*,|\s*\.|$)', text)
    experience_years = re.findall(r'(\d+)[\s\+]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', text_lower)
    return {'technical_skills': found_keywords, 'companies': [c.strip() for c in companies], 'experience_years': experience_years}

def extract_projects(text):
    projects = re.findall(r'(?:project:|worked on)\s*([A-Z][a-zA-Z0-9\s&-]+)', text)
    return [p.strip() for p in projects]

# New AI Rating Functions
async def generate_answer_review_and_rating(question, answer, position, experience_level):
    """Generate AI review and rating for a given answer"""
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
        
        Be strict with ratings:
        - Award 5 stars ONLY for truly exceptional answers that go above and beyond
        - Most good answers should be 3-4 stars maximum
        - Don't be generous - real interviewers are tough
        - Focus on technical accuracy, completeness, and depth
        
        For {experience_level} level candidates, expect:
        - Entry: Basic concepts and syntax knowledge
        - Mid: Practical experience and best practices  
        - Senior: Deep understanding, architecture decisions, trade-offs
        
        Provide:
        1. A critical review (2-3 sentences) pointing out what's missing or could be improved
        2. A strict rating based on real interview standards
        
        Format your response as:
        RATING: [1-5]
        REVIEW: [Your critical review focusing on gaps and improvements needed]
        """
        
        response = llm.invoke([HumanMessage(content=review_prompt)])
        if response and response.content:
            content = response.content.strip()
            
            # Extract rating
            rating_match = re.search(r'RATING:\s*(\d)', content)
            rating = int(rating_match.group(1)) if rating_match else 2  # Default to 2 instead of 3 for stricter evaluation
            
            # Extract review
            review_match = re.search(r'REVIEW:\s*(.*)', content, re.DOTALL)
            review = review_match.group(1).strip() if review_match else "Answer needs significant improvement to meet industry standards."
            
            return {"rating": rating, "review": review}
        
        return {"rating": 2, "review": "Answer requires substantial improvement to meet professional interview standards."}
        
    except Exception as e:
        print(f"Error generating review and rating: {e}")
        return {"rating": 2, "review": "Unable to generate detailed review. Answer needs significant improvement."}

async def calculate_hiring_probability(asked_questions, position, experience_level, total_questions_available):
    """Calculate hiring probability based on all answers, ratings, and completion rate"""
    try:
        if not asked_questions:
            return {"probability": 0, "feedback": "No questions answered. Interview incomplete.", "average_rating": 0}
        
        # Calculate completion rate
        completion_rate = len(asked_questions) / max(total_questions_available, 1)
        
        # Early termination penalty - heavily penalize incomplete interviews
        if completion_rate < 0.3:  # Less than 30% completed
            return {
                "probability": min(5, int(completion_rate * 100 * 0.2)),  # Max 5% for very incomplete interviews
                "feedback": f"Interview significantly incomplete ({len(asked_questions)}/{total_questions_available} questions answered). Insufficient data to make a hiring decision. Completion rate: {completion_rate:.1%}. Recommend completing at least 30% of questions for meaningful evaluation.",
                "average_rating": 0,
                "completion_rate": completion_rate
            }
        
        # Calculate average rating
        ratings = [q.get('rating', 1) for q in asked_questions]  # Default to 1 instead of 3 for failed questions
        avg_rating = statistics.mean(ratings)
        
        # Create summary of performance
        total_answered = len(asked_questions)
        excellent_count = len([r for r in ratings if r >= 4])
        good_count = len([r for r in ratings if r == 3])
        poor_count = len([r for r in ratings if r <= 2])
        
        # Calculate base probability from performance (stricter thresholds)
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
        completion_multiplier = min(1.0, completion_rate + 0.3)  # Bonus for completion, but cap at 1.0
        if completion_rate < 0.5:  # Less than 50% completed
            completion_multiplier *= 0.6  # Additional penalty
        elif completion_rate < 0.7:  # Less than 70% completed
            completion_multiplier *= 0.8  # Moderate penalty
        
        # Calculate final probability
        final_probability = int(base_probability * completion_multiplier)
        final_probability = max(0, min(100, final_probability))  # Ensure 0-100 range
        
        # Generate detailed feedback
        hiring_prompt = f"""
        You are an HR manager evaluating a candidate for the position of "{position}" at {experience_level} level.
        
        Interview Performance Summary:
        - Questions answered: {total_answered}/{total_questions_available} ({completion_rate:.1%} completion)
        - Average rating: {avg_rating:.1f}/5.0
        - Excellent answers (4-5 stars): {excellent_count}
        - Good answers (3 stars): {good_count}
        - Poor answers (1-2 stars): {poor_count}
        - Calculated hiring probability: {final_probability}%
        
        Provide detailed feedback explaining why this hiring probability is appropriate, considering:
        1. The completion rate impact
        2. Answer quality
        3. Overall interview performance
        4. What the candidate should improve
        
        Be realistic about hiring chances - incomplete interviews should have low probabilities.
        
        Format your response as:
        FEEDBACK: [Your detailed explanation]
        """
        
        try:
            response = llm.invoke([HumanMessage(content=hiring_prompt)])
            if response and response.content:
                content = response.content.strip()
                feedback_match = re.search(r'FEEDBACK:\s*(.*)', content, re.DOTALL)
                ai_feedback = feedback_match.group(1).strip() if feedback_match else "Standard evaluation completed."
            else:
                ai_feedback = "Standard evaluation completed."
        except:
            ai_feedback = f"Interview {completion_rate:.1%} complete with average rating of {avg_rating:.1f}/5.0."
        
        # Add completion rate context to feedback
        if completion_rate < 0.5:
            completion_context = f" Interview was only {completion_rate:.1%} complete ({total_answered}/{total_questions_available} questions), which significantly impacts hiring potential."
            ai_feedback = completion_context + " " + ai_feedback
        
        return {
            "probability": final_probability, 
            "feedback": ai_feedback, 
            "average_rating": round(avg_rating, 1),
            "completion_rate": completion_rate,
            "questions_answered": total_answered,
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
async def upload_resume(file: UploadFile = File(...), session_id: str = "default"):
    try:
        # Read file content
        file_content = await file.read()
        
        # Process based on file type
        if file.filename.endswith('.pdf'):
            text = read_pdf_from_bytes(file_content)
        elif file.filename.endswith('.docx'):
            text = read_docx_from_bytes(file_content)
        elif file.filename.endswith('.txt'):
            text = file_content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Process resume
        resume_chunks = chunk_resume(text)
        resume_keywords = extract_keywords(text)
        resume_projects = extract_projects(text)
        
        # Store in session
        if session_id not in sessions:
            sessions[session_id] = SessionData()
        
        sessions[session_id].resume_chunks = resume_chunks
        sessions[session_id].resume_keywords = resume_keywords
        sessions[session_id].resume_projects = resume_projects
        
        return {
            "success": True,
            "message": f"Resume processed successfully. Found {len(resume_chunks)} chunks and {len(resume_projects)} projects.",
            "chunks_count": len(resume_chunks),
            "projects_count": len(resume_projects),
            "keywords": resume_keywords
        }
    
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Error setting up interview: {str(e)}")

async def generate_questions(category, position, experience_level, resume_chunks):
    try:
        prompt = f"""
        You are an AI Interview Question Generator.
        Based on the following resume chunks, generate 6 {category} interview questions for the role of '{position}' suitable for a(n) {experience_level} level candidate.
        
        Resume Content:
        {''.join(resume_chunks)}
        
        Only return the questions in a numbered list format.
        Make sure to ask technical questions, they should be based on the level of experience.
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
        return 'relevant'  # Default to relevant if assessment fails
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
            len(session.question_queue)  # Pass total questions available
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)