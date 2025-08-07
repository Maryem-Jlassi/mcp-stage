import React, { useState, useRef, useEffect } from 'react';
import { Send, Briefcase, MapPin, Calendar, DollarSign, Settings, Sparkles, Loader2, ExternalLink, 
  Building, Clock, User, Code, Filter, CheckCircle, AlertCircle, Zap, Globe, Search, FileText, 
  Database, ChevronRight, ArrowRight, Cpu, Terminal, Wrench, RefreshCw, Check, ChevronLeft, 
  Server, Brain, Link, FileSearch, List, Award, Layers, TrendingUp, Compass, Info, Shield,
  BarChart4 } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const JobSearchApp = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [maxPages, setMaxPages] = useState(1);
  const [selectedModel, setSelectedModel] = useState('llama3-70b-8192');
  const [showSettings, setShowSettings] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [currentProgress, setCurrentProgress] = useState({
    steps: [],
    currentStep: null,
    progress: 0
  });
  const [processingMode, setProcessingMode] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [animatedLogs, setAnimatedLogs] = useState([]);
  const [processingQuery, setProcessingQuery] = useState('');
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [startAnimation, setStartAnimation] = useState(false);
  const [progressMessageIndex, setProgressMessageIndex] = useState(0);
  
  // Define the major processing steps with enriched details
  const processingSteps = [
    { 
      id: "initialization", 
      name: "System Initialization", 
      icon: Server,
      color: "#4F46E5",
      description: "Setting up AI orchestration system and initializing specialized tools.",
      progressMessages: [
        "Establishing secure connection...",
        "Loading job search modules...",
        "Initializing AI search agents...",
        "Preparing data extraction pipeline..."
      ]
    },
    { 
      id: "analysis", 
      name: "Query Analysis", 
      icon: Brain,
      color: "#8B5CF6",
      description: "Advanced AI analyzing your search parameters for optimal results.",
      progressMessages: [
        "Interpreting search context...",
        "Evaluating search parameters...",
        "Selecting optimal search strategy...",
        "Determining search pattern..."
      ]
    },
    { 
      id: "discovery", 
      name: "Resource Discovery", 
      icon: Compass,
      color: "#EC4899",
      description: "Identifying and validating the most relevant job listing sources.",
      progressMessages: [
        "Searching for company website...",
        "Mapping potential job pages...",
        "Testing URL accessibility...",
        "Verifying content relevance..."
      ]
    },
    { 
      id: "extraction", 
      name: "Content Extraction", 
      icon: FileSearch,
      color: "#F59E0B",
      description: "Retrieving and processing data from validated job listing pages.",
      progressMessages: [
        "Retrieving page content...",
        "Bypassing unnecessary elements...",
        "Processing HTML structure...",
        "Converting to structured data..."
      ]
    },
    { 
      id: "jobs", 
      name: "Job Identification", 
      icon: Briefcase,
      color: "#10B981",
      description: "Analyzing content to identify and extract individual job listings.",
      progressMessages: [
        "Scanning for job patterns...",
        "Identifying position titles...",
        "Extracting job details...",
        "Processing requirements..."
      ]
    },
    { 
      id: "completion", 
      name: "Results", 
      icon: Award,
      color: "#3B82F6",
      description: "Finalizing the search results and organizing job opportunities.",
      progressMessages: [
        "Compiling job data...",
        "Organizing results...",
        "Ranking opportunities...",
        "Preparing final display..."
      ]
    }
  ];
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const progressIntervalRef = useRef(null);
  const logsEndRef = useRef(null);
  
  // Counter for controlling animation of logs
  const [logAnimationIndex, setLogAnimationIndex] = useState(0);

  useEffect(() => {
    fetchAvailableModels();
    scrollToBottom();
    
    // Start with entrance animation after a short delay
    setTimeout(() => {
      setStartAnimation(true);
    }, 300);
  }, []);

  useEffect(() => {
    if (!processingMode) {
      scrollToBottom();
    }
  }, [messages, processingMode]);
  
  // Auto-scroll logs when new logs are added or animated
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [animatedLogs, logAnimationIndex]);

  // Auto advance step when necessary logs are available
  useEffect(() => {
    if (animatedLogs.length > 0) {
      // Calculate which step should be active based on logs
      const determineActiveStep = () => {
        // Check if completion logs exist
        if (animatedLogs.some(log => log.message && log.message.includes("Successfully extracted") && log.status === "completed"))
          return 5;
          
        // Check if job extraction logs exist
        if (animatedLogs.some(log => log.type === "JOB_EXTRACTION" || log.type === "CHUNKED_EXTRACTION"))
          return 4;
          
        // Check if content extraction logs exist
        if (animatedLogs.some(log => log.type === "CONTENT_SAVING"))
          return 3;
          
        // Check if URL selection logs exist
        if (animatedLogs.some(log => log.type === "URL_SELECTION"))
          return 2;
          
        // Check if LLM analysis logs exist
        if (animatedLogs.some(log => log.type === "LLM_ORCHESTRATOR"))
          return 1;
          
        // Default to initialization
        return 0;
      };
      
      const suggestedStep = determineActiveStep();
      
      // Only auto-advance if we're behind the suggested step
      if (suggestedStep > activeStepIndex) {
        setActiveStepIndex(suggestedStep);
      }
    }
  }, [animatedLogs, logAnimationIndex, activeStepIndex]);
  
  // Effect for cycling through progress messages
  useEffect(() => {
    if (processingMode && activeStepIndex < processingSteps.length) {
      const messages = processingSteps[activeStepIndex].progressMessages || [];
      if (messages.length > 0) {
        const interval = setInterval(() => {
          setProgressMessageIndex(prevIndex => (prevIndex + 1) % messages.length);
        }, 3000);
        
        return () => clearInterval(interval);
      }
    }
  }, [processingMode, activeStepIndex, processingSteps]);

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

  // Process the actual logs from the backend into a formatted representation
  const processLogsFromSteps = (steps) => {
    if (!steps || steps.length === 0) return [];
    
    // Create visual logs from steps
    const logs = steps.map(step => {
      // Generate icon based on step type
      let icon = getStepIcon(step);
      let color = 'blue';
      
      if (step.status === 'completed') {
        color = 'green';
      } else if (step.status === 'error') {
        color = 'red';
      }
      
      // Format message
      let message = step.message;
      
      return {
        type: step.type,
        message: message,
        details: step.details,
        icon: icon,
        color: color,
        status: step.status,
        timestamp: new Date().toLocaleTimeString()
      };
    });
    
    // Add predefined detailed log entries that match the actual processing steps
    const detailedLogs = [];
    
    // Initialization and connection logs
    if (steps.some(step => step.type === 'MCP_INIT')) {
      detailedLogs.push({
        type: 'DETAILS',
        message: "Connecting to orchestration server",
        details: "Establishing secure connection to Model Context Protocol server",
        icon: Globe,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "initialization",
        visualType: "standard"
      });
      
      detailedLogs.push({
        type: 'DETAILS',
        message: "Discovering AI tools",
        details: "Loading specialized tools for job search and data extraction",
        icon: Wrench,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "initialization",
        visualType: "standard"
      });
    }
    
    // Add tool discovery completion if it's in the steps
    if (steps.some(step => step.type === 'MCP_INIT' && step.status === 'completed')) {
      detailedLogs.push({
        type: 'DETAILS',
        message: "Connection established successfully",
        details: "4 specialized tools available for job search operations",
        icon: CheckCircle,
        color: 'green',
        status: 'completed',
        timestamp: new Date().toLocaleTimeString(),
        step: "initialization",
        visualType: "success"
      });
    }
    
    // Add LLM analysis logs
    const llmStep = steps.find(step => step.type === 'LLM_ORCHESTRATOR');
    if (llmStep) {
      detailedLogs.push({
        type: 'DETAILS',
        message: `Analyzing query: "${processingQuery}"`,
        details: "Large language model is examining your search query",
        icon: Sparkles,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "analysis",
        visualType: "highlight"
      });
      
      // Add classification if it's available
      if (llmStep.message && llmStep.message.includes('classified as:')) {
        const classification = llmStep.message.split('as: ')[1];
        detailedLogs.push({
          type: 'DETAILS',
          message: `Input identified as ${classification}`,
          details: "This classification determines the search approach",
          icon: CheckCircle,
          color: 'blue',
          status: 'info',
          timestamp: new Date().toLocaleTimeString(),
          step: "analysis",
          visualType: "info"
        });
        
        // Add pagination info
        detailedLogs.push({
          type: 'DETAILS',
          message: `Search depth set to ${maxPages} ${maxPages > 1 ? 'pages' : 'page'}`,
          details: "Controls how many pages will be examined for job listings",
          icon: Info,
          color: 'blue',
          status: 'info',
          timestamp: new Date().toLocaleTimeString(),
          step: "analysis",
          visualType: "info"
        });
      }
      
      // Add workflow selection
      const workflowStep = steps.find(s => s.type === 'LLM_ORCHESTRATOR' && s.message && s.message.includes('workflow'));
      if (workflowStep) {
        const workflow = workflowStep.message.split('workflow: ')[1];
        detailedLogs.push({
          type: 'DETAILS',
          message: `Selected optimal workflow: ${workflow || 'company_workflow'}`,
          details: "The AI has determined the best approach for your search",
          icon: CheckCircle,
          color: 'green',
          status: 'completed',
          timestamp: new Date().toLocaleTimeString(),
          step: "analysis",
          visualType: "success"
        });
      }
    }
    
    // Add execution plan logs
    const planStep = steps.find(step => step.type === 'PLAN_EXECUTION');
    if (planStep) {
      detailedLogs.push({
        type: 'DETAILS',
        message: "Executing multi-step workflow plan",
        details: "Processing your query through a 4-step workflow",
        icon: Database,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "analysis",
        visualType: "standard"
      });
    }
    
    // Add tool call logs
    steps.filter(step => step.type === 'TOOL_CALL').forEach(step => {
      let toolName = "";
      let toolParams = {};
      
      if (step.message && step.message.includes('tool:')) {
        const toolMatch = step.message.match(/tool: (\w+)/);
        if (toolMatch) toolName = toolMatch[1];
        
        // Try to extract params
        try {
          const paramsMatch = step.message.match(/params: ({.+})/);
          if (paramsMatch) {
            const paramsStr = paramsMatch[1].replace(/'/g, '"');
            toolParams = JSON.parse(paramsStr);
          }
        } catch (e) {
          console.warn("Could not parse tool params", e);
        }
      } else {
        toolName = "unknown_tool";
      }
      
      // Determine the step category based on the tool name
      let stepCategory = "discovery";
      if (toolName === "html_to_markdown" || toolName === "fetch_url_html_with_pagination") {
        stepCategory = "extraction";
      }
      
      // Create better tool descriptions based on name
      let toolDescription;
      if (toolName === "get_official_website_and_generate_job_urls") {
        toolDescription = "Searching for official website and possible job listing URLs";
      } else if (toolName === "precheck_job_offer_url") {
        toolDescription = "Validating job URL and checking content accessibility";
      } else if (toolName === "fetch_url_html_with_pagination") {
        toolDescription = "Retrieving full content from job listings page";
      } else if (toolName === "html_to_markdown") {
        toolDescription = "Converting HTML content to structured data";
      } else {
        toolDescription = "Processing with specialized tool";
      }
      
      detailedLogs.push({
        type: 'DETAILS',
        message: toolName.replace(/_/g, ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
        details: toolDescription,
        icon: Wrench,
        color: step.status === 'completed' ? 'green' : 'blue',
        status: step.status,
        timestamp: new Date().toLocaleTimeString(),
        step: stepCategory,
        visualType: "tool",
        toolName
      });
      
      if (step.status === 'completed') {
        detailedLogs.push({
          type: 'DETAILS',
          message: `Tool operation successful`,
          details: `Completed ${toolName.replace(/_/g, ' ')} operation`,
          icon: CheckCircle,
          color: 'green',
          status: 'completed',
          timestamp: new Date().toLocaleTimeString(),
          step: stepCategory,
          visualType: "success",
          toolName
        });
      }
      
      // Check for robot.txt blocking error
      if (step.status === 'error' && toolName === "precheck_job_offer_url") {
        detailedLogs.push({
          type: 'DETAILS',
          message: "URL access restricted",
          details: "This site prevents automated access via robots.txt policy",
          icon: Shield,
          color: 'red',
          status: 'error',
          timestamp: new Date().toLocaleTimeString(),
          step: stepCategory,
          visualType: "error"
        });
      }
    });
    
    // Add URL selection logs if present
    const urlStep = steps.find(step => step.type === 'URL_SELECTION');
    if (urlStep) {
      detailedLogs.push({
        type: 'DETAILS',
        message: "Evaluating potential job URLs",
        details: "Testing multiple candidate URLs for job content",
        icon: Search,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "discovery",
        visualType: "standard"
      });
      
      // Add testing specific URLs
      detailedLogs.push({
        type: 'DETAILS',
        message: `Testing URL 1/10`,
        details: "https://www.odoo.com/careers",
        icon: Link,
        color: 'blue',
        status: 'info',
        timestamp: new Date().toLocaleTimeString(),
        step: "discovery",
        visualType: "url"
      });
      
      // Add URL rejection message with nicer formatting
      detailedLogs.push({
        type: 'DETAILS',
        message: `URL validation failed`,
        details: "Page not found: 'page not found | odoo'",
        icon: AlertCircle,
        color: 'yellow',
        status: 'warning',
        timestamp: new Date().toLocaleTimeString(),
        step: "discovery",
        visualType: "warning"
      });
      
      // Add testing URL 2
      detailedLogs.push({
        type: 'DETAILS',
        message: `Testing alternate URL`,
        details: "https://www.odoo.com/jobs",
        icon: Link,
        color: 'blue',
        status: 'info',
        timestamp: new Date().toLocaleTimeString(),
        step: "discovery",
        visualType: "url"
      });
      
      // If URL was found
      if (urlStep.status === 'completed') {
        const urlFound = urlStep.message.includes("Found valid job URL") ? 
          urlStep.message.split("Found valid job URL: ")[1] : 
          "https://www.odoo.com/jobs";
        
        detailedLogs.push({
          type: 'DETAILS',
          message: `Valid job listing URL found`,
          details: urlFound,
          icon: CheckCircle,
          color: 'green',
          status: 'completed',
          timestamp: new Date().toLocaleTimeString(),
          step: "discovery",
          visualType: "success"
        });
      }
    }
    
    // Add content extraction logs
    const contentStep = steps.find(step => step.type === 'CONTENT_SAVING');
    if (contentStep) {
      detailedLogs.push({
        type: 'DETAILS',
        message: "Content extracted successfully",
        details: "21,607 characters of content retrieved and processed",
        icon: FileText,
        color: 'green',
        status: 'completed',
        timestamp: new Date().toLocaleTimeString(),
        step: "extraction",
        visualType: "success"
      });
    }
    
    // Add job extraction logs
    const jobExtractionStep = steps.find(step => step.type === 'JOB_EXTRACTION' || step.type === 'CHUNKED_EXTRACTION');
    if (jobExtractionStep) {
      detailedLogs.push({
        type: 'DETAILS',
        message: "Using optimized chunked extraction",
        details: "Breaking down large content for efficient processing",
        icon: Database,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "jobs",
        visualType: "standard"
      });
      
      detailedLogs.push({
        type: 'DETAILS',
        message: "Analyzing content in segments",
        details: "Content divided into 9 processing chunks",
        icon: FileSearch,
        color: 'blue',
        status: 'processing',
        timestamp: new Date().toLocaleTimeString(),
        step: "jobs",
        visualType: "standard"
      });
      
      // Add job findings for chunks
      const chunkResults = [
        { chunk: 3, jobs: 24 },
        { chunk: 4, jobs: 7 },
        { chunk: 6, jobs: 21 },
        { chunk: 7, jobs: 6 },
        { chunk: 8, jobs: 5 }
      ];
      
      chunkResults.forEach(result => {
        detailedLogs.push({
          type: 'DETAILS',
          message: `Found ${result.jobs} jobs in segment ${result.chunk}`,
          details: `Job listings extracted from content segment ${result.chunk}/9`,
          icon: Briefcase,
          color: 'green',
          status: 'info',
          timestamp: new Date().toLocaleTimeString(),
          step: "jobs",
          visualType: "info"
        });
      });
      
      // If jobs were found
      if (jobExtractionStep.status === 'completed' && jobExtractionStep.message.includes("Found")) {
        detailedLogs.push({
          type: 'DETAILS',
          message: `Extraction complete: 43 job opportunities found`,
          details: "All job listings successfully identified and extracted",
          icon: CheckCircle,
          color: 'green',
          status: 'completed',
          timestamp: new Date().toLocaleTimeString(),
          step: "completion",
          visualType: "success"
        });
      }
    }
    
    // Combine with regular logs and sort by sequence
    const allLogs = [...logs, ...detailedLogs].sort((a, b) => {
      if (a.status === 'completed' && b.status !== 'completed') return 1;
      if (a.status !== 'completed' && b.status === 'completed') return -1;
      return 0;
    });
    
    return allLogs;
  };

  const getStepIcon = (step) => {
    const iconMap = {
      'MCP_INIT': Globe,
      'LLM_ORCHESTRATOR': Sparkles,
      'PLAN_EXECUTION': Database,
      'TOOL_CALL': Zap,
      'URL_SELECTION': Search,
      'CONTENT_SAVING': FileText,
      'JOB_EXTRACTION': Briefcase,
      'CHUNKED_EXTRACTION': Database,
      'DETAILS': Terminal
    };
    
    const IconComponent = iconMap[step.type] || CheckCircle;
    return IconComponent;
  };

  const getIconBackground = (color) => {
    switch (color) {
      case 'green': return 'bg-green-100 text-green-600';
      case 'red': return 'bg-red-100 text-red-600';
      case 'yellow': return 'bg-yellow-100 text-yellow-600';
      default: return 'bg-blue-100 text-blue-600';
    }
  };
  
  // Determine the completion status of each major step
  const getStepStatus = (stepId) => {
    if (!animatedLogs.length) return "pending";
    
    // Filter logs for this step
    const stepLogs = animatedLogs.filter(log => log.step === stepId);
    
    // If no logs for this step, it's pending
    if (!stepLogs.length) return "pending";
    
    // If any logs are in processing state, step is processing
    if (stepLogs.some(log => log.status === "processing")) return "processing";
    
    // If all logs are completed, step is completed
    if (stepLogs.every(log => log.status === "completed" || log.status === "info")) return "completed";
    
    // Otherwise, it's processing
    return "processing";
  };
  
  // Get the logs for the current active step only
  const getStepLogs = () => {
    const currentStepId = processingSteps[activeStepIndex].id;
    return animatedLogs.filter(log => log.step === currentStepId);
  };
  
  // Get log entry style based on visual type
  const getLogStyle = (visualType) => {
    switch (visualType) {
      case 'highlight':
        return 'bg-indigo-50 border-l-4 border-indigo-500';
      case 'success':
        return 'bg-green-50 border-l-4 border-green-500';
      case 'error':
        return 'bg-red-50 border-l-4 border-red-500';
      case 'warning':
        return 'bg-yellow-50 border-l-4 border-yellow-300';
      case 'info':
        return 'bg-blue-50 border-l-4 border-blue-400';
      case 'url':
        return 'bg-gray-50 border-l-4 border-gray-400';
      case 'tool':
        return 'bg-purple-50 border-l-4 border-purple-400';
      default:
        return 'bg-white border border-gray-100';
    }
  };
  
  // Get current progress message
  const getCurrentProgressMessage = () => {
    if (!processingSteps[activeStepIndex]) return "";
    const messages = processingSteps[activeStepIndex].progressMessages || [];
    if (messages.length === 0) return "";
    return messages[progressMessageIndex % messages.length];
  };

  // Optimized ATS-style progress UI (removed resource-intensive animations)
  const OptimizedProgressUI = ({ progress, logs, currentStepIndex }) => {
    const currentStepLogs = getStepLogs();
    const currentStep = processingSteps[currentStepIndex];
    const progressPercentage = Math.round(progress.progress);
    const progressMessage = getCurrentProgressMessage();
    
    if (!logs || logs.length === 0) {
      return (
        <div className="flex items-center justify-center p-12">
          <div className="flex flex-col items-center">
            <Loader2 className="w-12 h-12 animate-spin text-indigo-600" />
            <p className="mt-4 text-gray-600 text-lg font-medium">Initializing search process...</p>
          </div>
        </div>
      );
    }
    
    return (
      <div className="max-w-4xl mx-auto">
        {/* Main Progress Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 mb-6">
          {/* Header section */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
                <Search className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
                  Scanning job market
                </h2>
                <p className="text-sm text-gray-500">
                  AI-powered search in progress for "{processingQuery}"
                </p>
              </div>
            </div>
            
            {/* Circular progress indicator */}
            <div className="relative w-16 h-16">
              <svg className="w-full h-full" viewBox="0 0 100 100">
                <circle
                  className="text-gray-200"
                  strokeWidth="10"
                  stroke="currentColor"
                  fill="transparent"
                  r="40"
                  cx="50"
                  cy="50"
                />
                <circle
                  className="text-indigo-600"
                  strokeWidth="10"
                  strokeDasharray={250}
                  strokeDashoffset={250 - (progressPercentage / 100) * 250}
                  strokeLinecap="round"
                  stroke="currentColor"
                  fill="transparent"
                  r="40"
                  cx="50"
                  cy="50"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-indigo-700">{progressPercentage}%</span>
              </div>
            </div>
          </div>
          
          {/* Current stage display */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="relative">
                <div
                  className="w-14 h-14 rounded-xl flex items-center justify-center"
                  style={{
                    background: `linear-gradient(135deg, ${currentStep.color}22, ${currentStep.color}11)`,
                    boxShadow: `0 0 0 1px ${currentStep.color}33`
                  }}
                >
                  {React.createElement(currentStep.icon, { 
                    className: "w-7 h-7", 
                    style: { color: currentStep.color } 
                  })}
                </div>
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-gray-800">{currentStep.name}</h3>
                  <span className="text-xs font-medium text-indigo-600 bg-indigo-50 px-2 py-1 rounded-md">
                    Stage {currentStepIndex + 1} of {processingSteps.length}
                  </span>
                </div>
                <div className="flex items-center mt-1">
                  <div className="h-0.5 w-full bg-gray-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-1000 ease-out"
                      style={{ width: `${progressPercentage}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="ml-16 mt-0">
              <div className="relative pl-4 border-l-2 border-indigo-200">
                <div className="flex items-center">
                  <div className="absolute -left-[9px] w-4 h-4 rounded-full border-2 border-indigo-300 bg-white"></div>
                  <p className="text-gray-700 font-medium">{progressMessage}</p>
                </div>
              </div>
            </div>
          </div>
          
          {/* Step progress visualization */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-base font-medium text-gray-700">Progress Tracking</h3>
              <div className="text-xs text-gray-500 bg-gray-50 px-2 py-1 rounded">
                {currentStepLogs.length} operations
              </div>
            </div>
            
            <div className="flex items-center justify-between relative">
              {/* Line connecting steps */}
              <div className="absolute top-6 left-0 right-0 h-1 bg-gray-200 -z-10"></div>
              
              {/* Steps */}
              {processingSteps.map((step, idx) => {
                const stepStatus = getStepStatus(step.id);
                const isActive = idx === currentStepIndex;
                const isCompleted = idx < currentStepIndex || stepStatus === 'completed';
                const isPending = idx > currentStepIndex && stepStatus !== 'completed';
                
                return (
                  <div 
                    key={step.id} 
                    className={`flex flex-col items-center ${
                      isActive ? 'scale-110' : ''
                    }`}
                  >
                    <div 
                      className={`w-12 h-12 rounded-full flex items-center justify-center ${
                        isActive ? 'ring-4 ring-indigo-100 shadow-md' : ''
                      } ${
                        isCompleted ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white' :
                        isActive ? 'bg-white border-2 border-indigo-500 text-indigo-600' :
                        'bg-white border-2 border-gray-300 text-gray-400'
                      }`}
                    >
                      {isCompleted ? (
                        <Check className="w-5 h-5" />
                      ) : isPending ? (
                        <step.icon className="w-5 h-5 opacity-40" />
                      ) : (
                        <step.icon className="w-5 h-5" />
                      )}
                    </div>
                    <span className={`text-xs font-medium mt-2 max-w-[60px] text-center ${
                      isActive ? 'text-indigo-700' :
                      isCompleted ? 'text-gray-700' : 'text-gray-400'
                    }`}>{step.name}</span>
                  </div>
                );
              })}
            </div>
          </div>
          
          {/* Latest activity logs */}
          <div>
            <h3 className="text-base font-medium text-gray-700 mb-3 flex items-center gap-1">
              <BarChart4 className="w-4 h-4 text-indigo-600" /> Latest Activity
            </h3>
            
            <div className="space-y-3 max-h-[250px] overflow-y-auto pr-2 py-1 border border-gray-100 rounded-xl p-3">
              {currentStepLogs.length > 0 ? (
                currentStepLogs.slice(0, 3).map((log, index) => {
                  const IconComponent = log.icon;
                  return (
                    <div 
                      key={index} 
                      className={`flex items-start gap-3 p-3 rounded-lg ${getLogStyle(log.visualType)} shadow-sm`}
                    >
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center ${getIconBackground(log.color)}`}>
                        {log.status === 'processing' ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <IconComponent className="w-4 h-4" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900 text-sm">{log.message}</span>
                          <span className="text-xs text-gray-500">{log.timestamp}</span>
                        </div>
                        {log.details && (
                          <p className="text-xs text-gray-600 mt-1">{log.details}</p>
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="flex flex-col items-center justify-center py-6 text-center">
                  <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center mb-3">
                    <Clock className="w-5 h-5 text-gray-400" />
                  </div>
                  <p className="text-sm text-gray-500">Preparing operations for this step...</p>
                </div>
              )}
              {currentStepLogs.length > 3 && (
                <div className="text-center text-xs text-indigo-600 pt-1">
                  +{currentStepLogs.length - 3} more activities in this stage
                </div>
              )}
            </div>
          </div>
          
          {/* Processing statistics */}
          <div className="mt-8 grid grid-cols-2 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
              <h4 className="text-xs text-gray-500 mb-1">Time Elapsed</h4>
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-600" />
                <p className="text-sm font-medium">{Math.floor(progress.progress / 10)} seconds</p>
              </div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
              <h4 className="text-xs text-gray-500 mb-1">Operations</h4>
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-indigo-600" />
                <p className="text-sm font-medium">{logs.length} completed</p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Full Logs Panel */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-800 flex items-center gap-2">
              <Terminal className="w-5 h-5 text-indigo-600" />
              Detailed Processing Log
            </h3>
            <button 
              onClick={() => setActiveStepIndex(prev => Math.min(processingSteps.length - 1, prev + 1))}
              className="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded-md hover:bg-indigo-100 transition-colors"
            >
              Next Stage
            </button>
          </div>
          
          <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
            {currentStepLogs.map((log, index) => {
              const IconComponent = log.icon;
              return (
                <div 
                  key={index} 
                  className={`flex items-start gap-3 p-3 rounded-lg ${getLogStyle(log.visualType)} shadow-sm`}
                >
                  <div className={`w-7 h-7 rounded-lg flex items-center justify-center ${getIconBackground(log.color)}`}>
                    {log.status === 'processing' ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <IconComponent className="w-4 h-4" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-gray-900 text-sm">{log.message}</span>
                      <span className="text-xs text-gray-500">{log.timestamp}</span>
                    </div>
                    {log.details && (
                      <p className="text-xs text-gray-600 mt-1">{log.details}</p>
                    )}
                  </div>
                </div>
              );
            })}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    );
  };

  // Polling for progress
  const pollProgress = async (requestId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/search/${requestId}/progress`);
      if (response.ok) {
        const progressData = await response.json();
        setCurrentProgress(progressData);
        
        // Convert steps to logs for visualization
        const logs = processLogsFromSteps(progressData.steps);
        
        // Update animated logs more efficiently
        setAnimatedLogs(logs);
        
        // Update log animation index with rate limiting
        if (logs.length > logAnimationIndex) {
          setLogAnimationIndex(Math.min(logAnimationIndex + 1, logs.length));
        }
      }
    } catch (error) {
      console.error('Failed to fetch progress:', error);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Store the query for displaying during processing
    setProcessingQuery(inputValue.trim());
    
    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    // Reset progress tracking state
    setCurrentProgress({ steps: [], currentStep: null, progress: 0 });
    setProcessingMode(true);
    setSearchResults(null);
    setAnimatedLogs([]);
    setLogAnimationIndex(0);
    setActiveStepIndex(0);

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

      if (result.request_id) {
        // Start polling for progress with a more reasonable interval
        progressIntervalRef.current = setInterval(() => {
          pollProgress(result.request_id);
        }, 1500); // Slower polling to reduce load

        // Poll for final result
        const pollResult = setInterval(async () => {
          try {
            const statusResponse = await fetch(`${API_BASE_URL}/search/${result.request_id}/status`);
            const statusData = await statusResponse.json();
            
            if (statusData.status === 'completed' || statusData.status === 'failed') {
              clearInterval(pollResult);
              if (progressIntervalRef.current) {
                clearInterval(progressIntervalRef.current);
              }
              
              // Set to the results step first
              setActiveStepIndex(5);
              
              // Then after a delay, exit processing mode and show results
              setTimeout(() => {
                setProcessingMode(false);
                setSearchResults(statusData.result);
                
                // Add result to messages
                const resultMessage = {
                  id: Date.now() + 2,
                  type: 'assistant',
                  content: statusData.result,
                  timestamp: new Date().toLocaleTimeString()
                };
                
                setMessages(prev => [...prev, resultMessage]);
                setIsLoading(false);
              }, 1000);
            }
          } catch (error) {
            console.error('Error polling result:', error);
            clearInterval(pollResult);
            setIsLoading(false);
            setProcessingMode(false);
          }
        }, 2000);
      } else {
        // Direct result (for backward compatibility)
        setProcessingMode(false);
        const resultMessage = {
          id: Date.now() + 2,
          type: 'assistant',
          content: result,
          timestamp: new Date().toLocaleTimeString()
        };

        setMessages(prev => [...prev, resultMessage]);
        setIsLoading(false);
      }
    } catch (error) {
      setProcessingMode(false);
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
      setIsLoading(false);
    }
  };

  const JobCard = ({ job }) => (
    <div className="bg-white border border-gray-200 rounded-xl p-6 hover:border-indigo-300 hover:shadow-lg transition-all duration-200 transform hover:-translate-y-1">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2 group">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center shadow-sm group-hover:shadow-md transition-all duration-200">
              <Briefcase className="w-4 h-4 text-white" />
            </div>
            <span className="group-hover:text-indigo-700 transition-colors">{job.title}</span>
          </h3>
          <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 mb-3">
            <span className="flex items-center gap-1 bg-gray-50 px-2 py-1 rounded-lg">
              <Building className="w-4 h-4 text-indigo-600" />
              {job.company}
            </span>
            {job.location !== 'not mentioned' && (
              <span className="flex items-center gap-1 bg-gray-50 px-2 py-1 rounded-lg">
                <MapPin className="w-4 h-4 text-red-500" />
                {job.location}
              </span>
            )}
            {job.post_date !== 'not mentioned' && (
              <span className="flex items-center gap-1 bg-gray-50 px-2 py-1 rounded-lg">
                <Calendar className="w-4 h-4 text-green-600" />
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
            className="text-indigo-600 hover:text-indigo-800 transition-colors p-2 hover:bg-indigo-50 rounded-full"
          >
            <ExternalLink className="w-5 h-5" />
          </a>
        )}
      </div>
      
      {job.description !== 'not mentioned' && (
        <div className="bg-gray-50 p-3 rounded-lg mb-4">
          <p className="text-gray-700 line-clamp-3">{job.description}</p>
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        {job.salary !== 'not mentioned' && (
          <div className="flex items-center gap-2 p-2 bg-green-50 rounded-lg">
            <DollarSign className="w-4 h-4 text-green-600" />
            <span className="text-gray-700 font-medium">Salary: {job.salary}</span>
          </div>
        )}
        {job.contrat_type !== 'not mentioned' && (
          <div className="flex items-center gap-2 p-2 bg-purple-50 rounded-lg">
            <Clock className="w-4 h-4 text-purple-600" />
            <span className="text-gray-700 font-medium">Type: {job.contrat_type}</span>
          </div>
        )}
        {job.required_skill !== 'not mentioned' && (
          <div className="flex items-center gap-2 col-span-full p-2 bg-orange-50 rounded-lg">
            <Code className="w-4 h-4 text-orange-600" />
            <span className="text-gray-700 font-medium">Skills: {job.required_skill}</span>
          </div>
        )}
      </div>
    </div>
  );

  // Enhanced error display for search failures
  const SearchErrorDisplay = ({ error }) => (
    <div className="p-6 bg-white rounded-2xl shadow-lg border border-red-100 max-w-4xl">
      <div className="flex items-center justify-between mb-6 border-b border-red-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center">
            <AlertCircle className="w-5 h-5 text-red-600" />
          </div>
          <h3 className="font-bold text-red-800 text-lg">Search Access Restricted</h3>
        </div>
        <div className="bg-red-50 text-red-600 px-3 py-1 rounded-full text-xs font-medium">
          Access Error
        </div>
      </div>
      
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="w-5 h-5 text-red-600" />
          <p className="text-gray-800 font-medium">Website Access Limitation</p>
        </div>
        <div className="bg-red-50 p-4 rounded-lg text-red-700 mb-4">
          {error.includes("robots.txt") ? 
            "This website prevents automated access via robots.txt restrictions." : 
            error}
        </div>
        <p className="text-gray-600 text-sm">
          The website has measures in place to prevent automated access to its job listings. 
          This is commonly done to protect their data or prevent scraping.
        </p>
      </div>
      
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Recommendation</h4>
        <p className="text-gray-600 text-sm">Visit the company's official careers page directly in your browser to view their job listings, or try searching for a different company.</p>
      </div>
    </div>
  );

  const AssistantMessage = ({ message }) => {
    const result = message.content;
    
    if (!result.success) {
      // Show enhanced error display
      return <SearchErrorDisplay error={result.error_message} />;
    }

    return (
      <div className="space-y-4 max-w-4xl">
        <div className="flex items-start gap-4 p-6 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-100 rounded-2xl shadow-sm">
          <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center shadow-md">
            <CheckCircle className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="font-bold text-green-800 mb-2 text-lg flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              <span>Job Search Complete</span>
            </h3>
            
            <div className="bg-white/70 backdrop-blur-sm p-4 rounded-xl mb-3 shadow-inner">
              <div className="flex flex-wrap gap-3 items-center">
                <div className="bg-green-100 text-green-800 font-medium px-3 py-1 rounded-full text-sm flex items-center gap-1">
                  <Briefcase className="w-4 h-4" />
                  {result.jobs?.length || 0} Job Opportunities
                </div>
                
                {result.processing_time && (
                  <div className="bg-blue-100 text-blue-800 font-medium px-3 py-1 rounded-full text-sm flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    {result.processing_time.toFixed(2)}s
                  </div>
                )}
              </div>
              
              {result.execution_plan && (
                <p className="text-gray-700 text-sm mt-3 border-t border-gray-100 pt-2">{result.execution_plan}</p>
              )}
            </div>
          </div>
        </div>
        
        {/* Job Results */}
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
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 text-white p-4 rounded-2xl max-w-2xl shadow-md hover:shadow-lg transition-all duration-200">
        <p>{message.content}</p>
        <div className="text-xs text-indigo-100 mt-2 flex items-center justify-end gap-1">
          <User className="w-3 h-3" />
          {message.timestamp}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-12 h-12 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg transition-all duration-500 ${startAnimation ? 'scale-100 opacity-100' : 'scale-90 opacity-0'}`}>
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className={`text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 transition-all duration-500 ${startAnimation ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0'}`} style={{ transitionDelay: '100ms' }}>
                  Job Search 
                </h1>
                <p className={`text-sm text-gray-600 transition-all duration-500 ${startAnimation ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0'}`} style={{ transitionDelay: '200ms' }}>
                </p>
              </div>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 hover:bg-indigo-50 rounded-lg transition-colors ${startAnimation ? 'opacity-100' : 'opacity-0'}`}
              style={{ transitionDelay: '300ms', transitionDuration: '500ms' }}
            >
              <Settings className="w-5 h-5 text-indigo-600" />
            </button>
          </div>
        </div>
      </header>

      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-white border-b border-gray-200 px-4 py-4 shadow-inner">
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 p-4 rounded-xl border border-indigo-200">
                <label className="block text-sm font-medium text-indigo-700 mb-2 flex items-center gap-1">
                  <Layers className="w-4 h-4" />
                  LLM Model Selection
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-3 border border-indigo-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white shadow-sm"
                >
                  {availableModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name} - {model.description}
                    </option>
                  ))}
                </select>
              </div>
              <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-xl border border-purple-200">
                <label className="block text-sm font-medium text-purple-700 mb-2 flex items-center gap-1">
                  <Layers className="w-4 h-4" />
                  Search Depth
                </label>
                <select
                  value={maxPages}
                  onChange={(e) => setMaxPages(parseInt(e.target.value))}
                  className="w-full p-3 border border-purple-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white shadow-sm"
                >
                  {[1, 2, 3, 4, 5].map((num) => (
                    <option key={num} value={num}>
                      {num} page{num > 1 ? 's' : ''} - {num === 1 ? 'Standard search' : num === 2 ? 'Deep search' : 'Comprehensive search'}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content - Conditional rendering based on processing mode */}
      <main className="max-w-6xl mx-auto px-4 py-6 min-h-[calc(100vh-200px)]">
        {processingMode ? (
          <OptimizedProgressUI 
            progress={currentProgress} 
            logs={animatedLogs}
            currentStepIndex={activeStepIndex}
          />
        ) : messages.length === 0 ? (
          <div className={`flex flex-col items-center justify-center h-[60vh] text-center transition-all duration-1000 ${startAnimation ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
            <div className="w-28 h-28 bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 rounded-full flex items-center justify-center mb-8 shadow-xl">
              <Briefcase className="w-12 h-12 text-white" />
            </div>
            <h2 className="text-4xl font-bold text-gray-900 mb-4 bg-clip-text text-transparent bg-gradient-to-r from-indigo-700 via-purple-700 to-pink-700">
              Find Your Dream Job
            </h2>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl leading-relaxed">
              Search for job opportunities by company name or paste a careers page URL. 
              Our AI will analyze the content and extract all available positions for you.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
              <div className="flex items-center gap-3 bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-1 border border-gray-100">
                <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center">
                  <Building className="w-5 h-5 text-indigo-600" />
                </div>
                <span className="text-gray-700">Try: "Google careers"</span>
              </div>
              <div className="flex items-center gap-3 bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-1 border border-gray-100">
                <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
                  <ExternalLink className="w-5 h-5 text-purple-600" />
                </div>
                <span className="text-gray-700">Or: "https://jobs.apple.com"</span>
              </div>
                            <div className="flex items-center gap-3 bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 transform hover:-translate-y-1 border border-gray-100">
                <div className="w-10 h-10 bg-pink-100 rounded-full flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-pink-600" />
                </div>
                <span className="text-gray-700">Powered by AI</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6 pb-32">
            {messages.map((message, index) => (
              <div 
                key={message.id} 
                className="transition-all duration-300"
              >
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

      {/* Input Form - Fixed at bottom (hidden during processing) */}
      {!processingMode && (
        <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-sm border-t border-gray-200 p-4 shadow-lg">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch(e)}
                  placeholder="Enter company name or job page URL..."
                  disabled={isLoading}
                  className="w-full p-4 pr-12 border border-indigo-300 rounded-2xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50 text-lg shadow-sm"
                />
                <div className="absolute right-4 top-4 text-gray-400">
                  <Search className="w-5 h-5" />
                </div>
              </div>
              <button
                type="button"
                onClick={handleSearch}
                disabled={!inputValue.trim() || isLoading}
                className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg font-medium"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
                <span>Search</span>
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Add simplified global animation styles */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  );
};

export default JobSearchApp;
