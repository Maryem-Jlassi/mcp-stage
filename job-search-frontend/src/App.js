import React, { useState, useRef, useEffect } from 'react';
import { Send, Briefcase, MapPin, Calendar, DollarSign, Settings, Sparkles, Loader2, ExternalLink, Building, Clock, User, Code, Filter } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const JobSearchApp = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [maxPages, setMaxPages] = useState(1);
  const [selectedModel, setSelectedModel] = useState('llama3-70b-8192');
  const [showSettings, setShowSettings] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchAvailableModels();
    scrollToBottom();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`);
      const data = await response.json();
      setAvailableModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Add loading message
    const loadingMessage = {
      id: Date.now() + 1,
      type: 'assistant',
      content: 'Searching for job opportunities...',
      isLoading: true,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          max_pages: maxPages,
          model: selectedModel
        }),
      });

      const result = await response.json();

      // Remove loading message and add result
      setMessages(prev => prev.filter(msg => !msg.isLoading));
      
      const resultMessage = {
        id: Date.now() + 2,
        type: 'assistant',
        content: result,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, resultMessage]);
    } catch (error) {
      // Remove loading message and add error
      setMessages(prev => prev.filter(msg => !msg.isLoading));
      
      const errorMessage = {
        id: Date.now() + 2,
        type: 'assistant',
        content: {
          success: false,
          error_message: `Failed to search jobs: ${error.message}`,
          jobs: []
        },
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const JobCard = ({ job }) => (
    <div className="bg-white border border-gray-200 rounded-xl p-6 hover:border-blue-300 hover:shadow-lg transition-all duration-200">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
            <Briefcase className="w-5 h-5 text-blue-600" />
            {job.title}
          </h3>
          <div className="flex items-center gap-4 text-sm text-gray-600 mb-3">
            <span className="flex items-center gap-1">
              <Building className="w-4 h-4" />
              {job.company}
            </span>
            {job.location !== 'not mentioned' && (
              <span className="flex items-center gap-1">
                <MapPin className="w-4 h-4" />
                {job.location}
              </span>
            )}
            {job.post_date !== 'not mentioned' && (
              <span className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                {job.post_date}
              </span>
            )}
          </div>
        </div>
        {job.url && (
          <a
            href={job.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 transition-colors"
          >
            <ExternalLink className="w-5 h-5" />
          </a>
        )}
      </div>
      
      {job.description !== 'not mentioned' && (
        <p className="text-gray-700 mb-4 line-clamp-3">{job.description}</p>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        {job.salary !== 'not mentioned' && (
          <div className="flex items-center gap-2">
            <DollarSign className="w-4 h-4 text-green-600" />
            <span className="text-gray-700">Salary: {job.salary}</span>
          </div>
        )}
        {job.contrat_type !== 'not mentioned' && (
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-purple-600" />
            <span className="text-gray-700">Type: {job.contrat_type}</span>
          </div>
        )}
        {job.required_skill !== 'not mentioned' && (
          <div className="flex items-center gap-2 col-span-full">
            <Code className="w-4 h-4 text-orange-600" />
            <span className="text-gray-700">Skills: {job.required_skill}</span>
          </div>
        )}
      </div>
    </div>
  );

  const AssistantMessage = ({ message }) => {
    const result = message.content;
    
    if (message.isLoading) {
      return (
        <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-2xl max-w-4xl">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
            <span className="text-gray-700">{message.content}</span>
          </div>
        </div>
      );
    }

    if (!result.success) {
      return (
        <div className="p-6 bg-red-50 border border-red-200 rounded-2xl max-w-4xl">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-red-800 mb-2">Search Failed</h3>
              <p className="text-red-700">{result.error_message}</p>
              {result.tools_used?.length > 0 && (
                <div className="mt-3 text-sm text-red-600">
                  Tools used: {result.tools_used.join(', ')}
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4 max-w-4xl">
        <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-2xl">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-green-800 mb-2">
              Found {result.jobs?.length || 0} Job Opportunities
            </h3>
            {result.execution_plan && (
              <p className="text-green-700 text-sm mb-2">{result.execution_plan}</p>
            )}
            <div className="flex items-center gap-4 text-xs text-green-600">
              {result.processing_time && (
                <span>⚡ {result.processing_time.toFixed(2)}s</span>
              )}
              {result.tools_used?.length > 0 && (
                <span>🔧 {result.tools_used.join(', ')}</span>
              )}
            </div>
          </div>
        </div>
        
        {result.jobs && result.jobs.length > 0 && (
          <div className="grid gap-4">
            {result.jobs.map((job, index) => (
              <JobCard key={index} job={job} />
            ))}
          </div>
        )}
      </div>
    );
  };

  const UserMessage = ({ message }) => (
    <div className="flex justify-end">
      <div className="bg-blue-600 text-white p-4 rounded-2xl max-w-2xl">
        <p>{message.content}</p>
        <div className="text-xs text-blue-100 mt-2">{message.timestamp}</div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Job Search AI</h1>
                <p className="text-sm text-gray-600">Powered by LLM Orchestration</p>
              </div>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <Settings className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>
      </header>

      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-white border-b border-gray-200 px-4 py-4">
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LLM Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {availableModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name} - {model.description}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Pages to Crawl
                </label>
                <select
                  value={maxPages}
                  onChange={(e) => setMaxPages(parseInt(e.target.value))}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {[1, 2, 3, 4, 5].map((num) => (
                    <option key={num} value={num}>
                      {num} page{num > 1 ? 's' : ''}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-6 min-h-[calc(100vh-200px)]">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[60vh] text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center mb-6">
              <Briefcase className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Find Your Dream Job
            </h2>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl">
              Search for job opportunities by company name or paste a careers page URL. 
              I'll analyze the content and extract all available positions for you.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <Building className="w-4 h-4" />
                <span>Try: "Google careers"</span>
              </div>
              <div className="flex items-center gap-2">
                <ExternalLink className="w-4 h-4" />
                <span>Or: "https://jobs.apple.com"</span>
              </div>
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4" />
                <span>AI-powered extraction</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6 pb-32">
            {messages.map((message) => (
              <div key={message.id}>
                {message.type === 'user' ? (
                  <UserMessage message={message} />
                ) : (
                  <AssistantMessage message={message} />
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Input Form - Fixed at bottom */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-sm border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSearch} className="flex gap-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Enter company name or job page URL..."
                disabled={isLoading}
                className="w-full p-4 pr-12 border border-gray-300 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 text-lg"
              />
            </div>
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default JobSearchApp;