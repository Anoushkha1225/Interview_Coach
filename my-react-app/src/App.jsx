import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Square, Mic, MicOff, FileText, Briefcase, User, ChevronRight, Clock, CheckCircle, XCircle, MessageSquare, BarChart3, Star, TrendingUp, Award } from 'lucide-react';

const InterviewBot = () => {
  const [currentStep, setCurrentStep] = useState('upload');
  const [resume, setResume] = useState(null);
  const [position, setPosition] = useState('');
  const [experienceLevel, setExperienceLevel] = useState('');
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [askedQuestions, setAskedQuestions] = useState([]);
  const [currentLevel, setCurrentLevel] = useState('Easy');
  const [processingAnswer, setProcessingAnswer] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState('default');
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [interviewSummary, setInterviewSummary] = useState(null);
  const fileInputRef = useRef(null);

  // Backend API base URL
  const API_BASE_URL = 'https://interview-coach-49a7.onrender.com';

  // Star Rating Component
  const StarRating = ({ rating, size = 'w-5 h-5' }) => {
    return (
      <div className="flex items-center space-x-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`${size} ${
              star <= rating ? 'text-yellow-400 fill-current' : 'text-gray-300'
            }`}
          />
        ))}
        <span className="ml-2 text-sm font-medium">{rating}/5</span>
      </div>
    );
  };

  // API call functions
  const uploadResume = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    const response = await fetch(`${API_BASE_URL}/upload-resume`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to upload resume');
    }

    return await response.json();
  };

  const setupInterview = async () => {
    const response = await fetch(`${API_BASE_URL}/setup-interview`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        position,
        experience_level: experienceLevel,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to setup interview');
    }

    return await response.json();
  };

  const getNextQuestion = async () => {
    const response = await fetch(`${API_BASE_URL}/get-next-question/${sessionId}`);
    
    if (!response.ok) {
      throw new Error('Failed to get next question');
    }

    return await response.json();
  };

  const submitAnswer = async (answer, question) => {
    const response = await fetch(`${API_BASE_URL}/submit-answer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        answer,
        question,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to submit answer');
    }

    return await response.json();
  };

  const getInterviewSummary = async () => {
    const response = await fetch(`${API_BASE_URL}/interview-summary/${sessionId}`);
    
    if (!response.ok) {
      throw new Error('Failed to get interview summary');
    }

    return await response.json();
  };

  // Event handlers
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setResume(file);
      setLoading(true);
      
      try {
        await uploadResume(file);
        setCurrentStep('setup');
      } catch (error) {
        console.error('Error uploading resume:', error);
        alert('Failed to upload resume. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleSetupComplete = async () => {
    if (position && experienceLevel) {
      setLoading(true);
      
      try {
        const setupResult = await setupInterview();
        setTotalQuestions(setupResult.total_questions);
        setCurrentStep('interview');
        
        const questionResult = await getNextQuestion();
        if (questionResult.success) {
          setCurrentQuestion({
            question: questionResult.question,
            level: questionResult.level
          });
          setCurrentLevel(questionResult.current_level);
        }
      } catch (error) {
        console.error('Error setting up interview:', error);
        alert('Failed to setup interview. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleAnswerSubmit = async () => {
    if (!userAnswer.trim() || !currentQuestion) return;
    
    setProcessingAnswer(true);
    
    try {
      const submitResult = await submitAnswer(userAnswer, currentQuestion.question);
      
      const newAskedQuestion = {
        level: currentQuestion.level,
        question: currentQuestion.question,
        answer: userAnswer,
        rating: submitResult.rating,
        review: submitResult.review,
        timestamp: new Date()
      };
      
      setAskedQuestions(prev => [...prev, newAskedQuestion]);
      setCurrentLevel(submitResult.new_level);
      setUserAnswer('');
      
      const nextQuestionResult = await getNextQuestion();
      
      if (nextQuestionResult.success && !nextQuestionResult.interview_complete) {
        setCurrentQuestion({
          question: nextQuestionResult.question,
          level: nextQuestionResult.level
        });
      } else {
        await handleInterviewComplete();
      }
      
    } catch (error) {
      console.error('Error submitting answer:', error);
      alert('Failed to submit answer. Please try again.');
    } finally {
      setProcessingAnswer(false);
    }
  };

  const handleInterviewComplete = async () => {
    try {
      const summary = await getInterviewSummary();
      setInterviewSummary(summary);
      setCurrentStep('complete');
    } catch (error) {
      console.error('Error getting interview summary:', error);
      setCurrentStep('complete');
    }
  };

  const handleEndInterview = async () => {
    await handleInterviewComplete();
  };

  const speakQuestion = (text) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.8;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    }
  };

  const resetInterview = async () => {
    try {
      await fetch(`${API_BASE_URL}/reset-session/${sessionId}`, {
        method: 'DELETE',
      });
    } catch (error) {
      console.error('Error resetting session:', error);
    }
    
    setCurrentStep('upload');
    setResume(null);
    setPosition('');
    setExperienceLevel('');
    setCurrentQuestion(null);
    setUserAnswer('');
    setAskedQuestions([]);
    setCurrentLevel('Easy');
    setTotalQuestions(0);
    setInterviewSummary(null);
  };

  const getHiringProbabilityColor = (probability) => {
    if (probability >= 70) return 'text-green-600 bg-green-50';
    if (probability >= 40) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getHiringProbabilityStatus = (probability) => {
    if (probability >= 70) return 'High';
    if (probability >= 40) return 'Moderate';
    return 'Low';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Interview Bot</h1>
          <p className="text-gray-600">AI-powered interview preparation with dynamic question adaptation</p>
          <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg max-w-2xl mx-auto">
            <p className="text-sm text-yellow-800">
              ⚠️ <strong>Practice Purpose Only:</strong> This is a simulation tool for interview preparation. 
              Ratings and assessments are for educational practice and do not represent actual hiring decisions.
            </p>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            {['upload', 'setup', 'interview', 'complete'].map((step, index) => (
              <div key={step} className="flex items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                  currentStep === step ? 'bg-blue-600 text-white' : 
                  ['upload', 'setup', 'interview', 'complete'].indexOf(currentStep) > index ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'
                }`}>
                  {index + 1}
                </div>
                {index < 3 && <ChevronRight className="w-5 h-5 text-gray-400 mx-2" />}
              </div>
            ))}
          </div>
        </div>

        {/* Upload Step */}
        {currentStep === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="text-center mb-6">
                <FileText className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Your Resume</h2>
                <p className="text-gray-600">Upload your resume in PDF, DOCX, or TXT format</p>
              </div>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={loading}
                />
                {loading ? (
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                    <p className="text-gray-600">Processing your resume...</p>
                  </div>
                ) : (
                  <>
                    <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 mb-4">Click to select or drag and drop your resume</p>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Choose File
                    </button>
                  </>
                )}
              </div>
              
              {resume && !loading && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
                  <div className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
                    <span className="text-green-800">Resume uploaded: {resume.name}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Setup Step */}
        {currentStep === 'setup' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="text-center mb-6">
                <Briefcase className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Interview Setup</h2>
                <p className="text-gray-600">Tell us about the position and your experience level</p>
              </div>

              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Position/Role
                  </label>
                  <input
                    type="text"
                    value={position}
                    onChange={(e) => setPosition(e.target.value)}
                    placeholder="e.g., Frontend Developer, Data Scientist, Product Manager"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Experience Level
                  </label>
                  <select
                    value={experienceLevel}
                    onChange={(e) => setExperienceLevel(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select your experience level</option>
                    <option value="entry">Entry Level (0-2 years)</option>
                    <option value="mid">Mid Level (2-5 years)</option>
                    <option value="senior">Senior Level (5+ years)</option>
                  </select>
                </div>

                <button
                  onClick={handleSetupComplete}
                  disabled={!position || !experienceLevel || loading}
                  className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Setting up interview...
                    </>
                  ) : (
                    'Start Interview'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Interview Step */}
        {currentStep === 'interview' && (
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Interview Panel */}
              <div className="lg:col-span-2">
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center space-x-4">
                      <MessageSquare className="w-6 h-6 text-blue-600" />
                      <h2 className="text-xl font-bold text-gray-800">Interview in Progress</h2>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        currentLevel === 'Easy' ? 'bg-green-100 text-green-800' :
                        currentLevel === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {currentLevel}
                      </span>
                      <button
                        onClick={handleEndInterview}
                        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
                      >
                        End Interview
                      </button>
                    </div>
                  </div>

                  {currentQuestion && (
                    <div className="space-y-6">
                      <div className="bg-blue-50 rounded-lg p-6">
                        <div className="flex items-center justify-between mb-4">
                          <h3 className="font-semibold text-gray-800">
                            Question {askedQuestions.length + 1} of {totalQuestions}
                          </h3>
                          <button
                            onClick={() => speakQuestion(currentQuestion.question)}
                            className="text-blue-600 hover:text-blue-700 p-2 rounded-lg hover:bg-blue-100 transition-colors"
                          >
                            <Play className="w-5 h-5" />
                          </button>
                        </div>
                        <p className="text-lg text-gray-800">{currentQuestion.question}</p>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Your Answer
                        </label>
                        <textarea
                          value={userAnswer}
                          onChange={(e) => setUserAnswer(e.target.value)}
                          placeholder="Type your answer here..."
                          rows={6}
                          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                        />
                      </div>

                      <div className="flex justify-end">
                        <button
                          onClick={handleAnswerSubmit}
                          disabled={!userAnswer.trim() || processingAnswer}
                          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                        >
                          {processingAnswer ? (
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                          ) : (
                            <ChevronRight className="w-5 h-5" />
                          )}
                          <span>{processingAnswer ? 'Processing...' : 'Next Question'}</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Progress */}
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <h3 className="font-bold text-gray-800 mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Progress
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span>Questions Answered</span>
                      <span>{askedQuestions.length}/{totalQuestions}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: totalQuestions > 0 ? `${(askedQuestions.length / totalQuestions) * 100}%` : '0%' }}
                      ></div>
                    </div>
                    <div className="text-sm text-gray-600">
                      Current Level: <span className="font-medium">{currentLevel}</span>
                    </div>
                  </div>
                </div>

                {/* Recent Questions with Ratings */}
                {askedQuestions.length > 0 && (
                  <div className="bg-white rounded-lg shadow-lg p-6">
                    <h3 className="font-bold text-gray-800 mb-4 flex items-center">
                      <Clock className="w-5 h-5 mr-2" />
                      Recent Questions
                    </h3>
                    <div className="space-y-4 max-h-60 overflow-y-auto">
                      {askedQuestions.slice(-3).reverse().map((q, index) => (
                        <div key={index} className="border-l-4 border-blue-300 pl-3 py-2">
                          <div className="flex items-center justify-between mb-2">
                            <span className={`text-xs font-medium ${
                              q.level === 'Easy' ? 'text-green-600' :
                              q.level === 'Medium' ? 'text-yellow-600' :
                              'text-red-600'
                            }`}>
                              {q.level}
                            </span>
                            <StarRating rating={q.rating} size="w-3 h-3" />
                          </div>
                          <p className="text-sm text-gray-700 line-clamp-2">{q.question}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Complete Step */}
        {currentStep === 'complete' && interviewSummary && (
          <div className="max-w-6xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="text-center mb-8">
                <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Interview Complete!</h2>
                <p className="text-gray-600">Here's your detailed performance analysis</p>
                <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg max-w-3xl mx-auto">
                  <p className="text-sm text-blue-800">
                    📊 <strong>Practice Assessment:</strong> This evaluation uses strict industry standards for realistic feedback. 
                    Results are for learning purposes and designed to help you improve your interview skills.
                  </p>
                </div>
              </div>

              {/* Overall Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{interviewSummary.questions_answered}</div>
                  <div className="text-sm text-gray-600">Questions Answered</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{Math.round((interviewSummary.completion_rate || 0) * 100)}%</div>
                  <div className="text-sm text-gray-600">Completion Rate</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-600 flex items-center justify-center">
                    <StarRating rating={interviewSummary.overall_rating || 0} size="w-6 h-6" />
                  </div>
                  <div className="text-sm text-gray-600">Overall Rating</div>
                </div>
                <div className={`text-center p-4 rounded-lg ${getHiringProbabilityColor(interviewSummary.hiring_probability)}`}>
                  <div className="text-2xl font-bold">{interviewSummary.hiring_probability}%</div>
                  <div className="text-sm">Hiring Probability</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">{interviewSummary.final_level}</div>
                  <div className="text-sm text-gray-600">Final Difficulty</div>
                </div>
              </div>

              {/* Hiring Assessment */}
              <div className="mb-8 p-6 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg border-l-4 border-indigo-500">
                <div className="flex items-center mb-4">
                  <TrendingUp className="w-6 h-6 text-indigo-600 mr-2" />
                  <h3 className="text-xl font-bold text-gray-800">Hiring Assessment</h3>
                  <span className={`ml-auto px-3 py-1 rounded-full text-sm font-medium ${getHiringProbabilityColor(interviewSummary.hiring_probability)}`}>
                    {getHiringProbabilityStatus(interviewSummary.hiring_probability)} Chance
                  </span>
                </div>
                <p className="text-gray-700 leading-relaxed">{interviewSummary.hiring_feedback}</p>
              </div>

              {/* Detailed Question Analysis */}
              {interviewSummary.asked_questions && interviewSummary.asked_questions.length > 0 && (
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                    <Award className="w-6 h-6 text-yellow-500 mr-2" />
                    Detailed Question Analysis
                  </h3>
                  <div className="space-y-6 max-h-96 overflow-y-auto">
                    {interviewSummary.asked_questions.map((q, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center space-x-3">
                            <span className="font-medium text-gray-800">Question {index + 1}</span>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              q.level === 'Easy' ? 'bg-green-100 text-green-800' :
                              q.level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              {q.level}
                            </span>
                          </div>
                          <StarRating rating={q.rating} />
                        </div>
                        
                        <div className="mb-4">
                          <h4 className="font-medium text-gray-800 mb-2">Question:</h4>
                          <p className="text-gray-700 bg-gray-50 p-3 rounded">{q.question}</p>
                        </div>
                        
                        <div className="mb-4">
                          <h4 className="font-medium text-gray-800 mb-2">Your Answer:</h4>
                          <p className="text-gray-700 bg-blue-50 p-3 rounded leading-relaxed">{q.answer}</p>
                        </div>
                        
                        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-lg border-l-4 border-yellow-400">
                          <h4 className="font-medium text-gray-800 mb-2 flex items-center">
                            <MessageSquare className="w-4 h-4 mr-2" />
                            AI Review (Strict Industry Standards):
                          </h4>
                          <p className="text-gray-700 leading-relaxed">{q.review}</p>
                          <div className="mt-2 text-xs text-gray-600 italic">
                            * This is practice feedback designed to help you improve
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-8 text-center space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600 mb-2">Ready for your next interview challenge?</p>
                  <button
                    onClick={resetInterview}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Start New Interview
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Fallback Complete Step (if no summary) */}
        {currentStep === 'complete' && !interviewSummary && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="text-center mb-8">
                <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Interview Complete!</h2>
                <p className="text-gray-600">Thank you for completing the interview session.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{askedQuestions.length}</div>
                  <div className="text-sm text-gray-600">Questions Answered</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{position}</div>
                  <div className="text-sm text-gray-600">Position</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">{experienceLevel}</div>
                  <div className="text-sm text-gray-600">Experience Level</div>
                </div>
              </div>

              <div className="mt-8 text-center">
                <button
                  onClick={resetInterview}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Start New Interview
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewBot;