#! /usr/bin/env python

import operator
import os
import asyncio
import time
import json
import logging
from typing import Annotated, TypedDict, Union, List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enhanced Web Browser Tools Setup ---

# Initialize Google Search with better error handling
search = GoogleSerperAPIWrapper()

@lru_cache(maxsize=100)
def google_search(query: str) -> str:
    """Enhanced Google search with caching and better error handling."""
    try:
        logger.info(f"Performing Google search for: {query[:50]}...")
        result = search.run(query)
        logger.info(f"Search completed successfully, result length: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {str(e)}")
        return f"Search failed: {str(e)}"

def web_browse(url: str) -> str:
    """Enhanced web browsing with better content extraction and error handling."""
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
        
        logger.info(f"Browsing URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content with better formatting
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Return first 3000 characters for better content
        if len(text) > 3000:
            text = text[:3000] + "..."
        
        logger.info(f"Successfully extracted {len(text)} characters from {url}")
        return text
        
    except Exception as e:
        logger.error(f"Failed to browse {url}: {str(e)}")
        return f"Failed to browse {url}: {str(e)}"

# Enhanced web tools with better descriptions
web_tools = [
    Tool(
        name="google_search",
        description="Search Google for current information, facts, news, or research. Use when you need up-to-date information, recent developments, or to verify facts.",
        func=google_search
    ),
    Tool(
        name="web_browse",
        description="Browse a specific web page to get detailed information from a URL. Use when you have a specific URL that contains relevant information.",
        func=web_browse
    )
]

# --- Enhanced Models with Better Configuration ---

# Model configurations with optimized parameters
model_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 4000,
    "stream": True
}

# Initialize models with better configurations
model_alice = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,
    top_p=0.9,
    streaming=True
)

model_bob = ChatOllama(
    model="llama3-2.3b:latest", 
    temperature=0.6,  # Slightly lower for technical accuracy
    top_p=0.9,
    streaming=True
)



model_supervisor = ChatOllama(
    model="llama3.2-3b-grpo:latest",
    temperature=0.5,  # Lower temperature for more consistent routing
    top_p=0.9,
    streaming=True
)


# model_alice = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-preview-04-17",
#     temperature=0.7,
#     top_p=0.9,

# )

# model_bob = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-preview-04-17", 
#     temperature=0.6, 
#     top_p=0.9,

# )



# model_supervisor = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-preview-04-17",
#     temperature=0.5,  # Lower temperature for more consistent routing
#     top_p=0.9,

# )

# --- Enhanced Difficulty Assessment ---

def is_difficult_question(question: str) -> Dict[str, Union[bool, str, List[str], int]]:
    """Enhanced difficulty assessment with detailed analysis."""
    difficult_keywords = [
        "current", "latest", "recent", "today", "2024", "2025", "news",
        "stock price", "weather", "events", "what happened", "when did",
        "latest version", "recent developments", "breaking news",
        "research", "study", "scientific", "technical", "complex",
        "detailed explanation", "in-depth", "comprehensive", "compare",
        "analysis", "statistics", "data", "trends", "forecast",
        "implementation", "code", "programming", "algorithm", "architecture"
    ]
    
    question_lower = question.lower()
    detected_keywords = []
    complexity_score = 0
    
    # Check for difficult keywords
    for keyword in difficult_keywords:
        if keyword in question_lower:
            detected_keywords.append(keyword)
            complexity_score += 1
    
    # Check for question complexity
    word_count = len(question.split())
    if word_count > 15:
        complexity_score += 2
    elif word_count > 10:
        complexity_score += 1
    
    # Check for multiple questions
    question_count = question.count('?')
    if question_count > 1:
        complexity_score += 2
    
    # Check for technical terms
    technical_terms = ["api", "sdk", "framework", "library", "database", "server", "client", "protocol"]
    for term in technical_terms:
        if term in question_lower:
            complexity_score += 1
            detected_keywords.append(term)
    
    # Determine if tools are needed
    use_tools = complexity_score >= 2
    
    return {
        "use_tools": use_tools,
        "complexity_score": complexity_score,
        "detected_keywords": detected_keywords,
        "word_count": word_count,
        "question_count": question_count
    }

# --- Enhanced Agent Prompts ---

# Alice: Enhanced General Knowledge Agent
alice_simple_prompt = PromptTemplate.from_template(
    """You are Alice, a General Knowledge Agent with PhD-level expertise across multiple disciplines. Your knowledge spans science, technology, humanities, arts, business, and more. You excel at providing comprehensive, well-reasoned responses that draw from diverse fields of knowledge.

IMPORTANT GUIDELINES:
- Provide accurate, well-structured responses
- Use clear, engaging language
- Include relevant examples and analogies
- Connect concepts across disciplines when relevant
- Acknowledge limitations when appropriate
- Focus on depth, accuracy, and interdisciplinary connections

Conversation History:
{chat_history}

User Request:
{input}

Your response:"""
)

alice_tool_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are Alice, a General Knowledge Agent with PhD-level expertise across multiple disciplines. 
You have comprehensive knowledge spanning science, technology, humanities, arts, business, and more.

RESPONSIBILITIES:
- Use web search tools to research current information and verify facts
- Provide thorough, well-reasoned responses with proper citations
- Demonstrate deep understanding and interdisciplinary thinking
- Ensure accuracy by cross-referencing information when possible
- Present information in a clear, engaging manner

When using tools:
- Search for the most relevant and current information
- Verify facts from multiple sources when possible
- Synthesize information from different sources
- Provide context and explain the significance of findings"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Bob: Enhanced Technical Implementation Agent
bob_simple_prompt = PromptTemplate.from_template(
    """You are Bob, a Technical Implementation Agent with deep expertise in software development, system architecture, and technical problem-solving.

EXPERTISE AREAS:
- Programming languages and frameworks
- System design and architecture
- Database design and optimization
- API development and integration
- DevOps and deployment
- Performance optimization
- Security best practices

RESPONSE GUIDELINES:
- Provide practical, implementable solutions
- Include code examples when relevant
- Explain technical concepts clearly
- Consider scalability and maintainability
- Address potential issues and edge cases
- Suggest best practices and alternatives

Conversation History:
{chat_history}

User Request:
{input}

Your response:"""
)

bob_tool_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are Bob, a Technical Implementation Agent with access to web search tools.

RESPONSIBILITIES:
- Research technical specifications and best practices
- Troubleshoot implementation challenges
- Provide accurate, efficient, and well-structured technical solutions
- Stay current with latest technologies and frameworks
- Verify technical information from authoritative sources

When using tools:
- Search for official documentation and specifications
- Look for best practices and community recommendations
- Verify version compatibility and requirements
- Find working examples and tutorials"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Enhanced Supervisor Prompt
supervisor_routing_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an intelligent supervisor managing a team of two specialized agents:

ALICE: General Knowledge Agent with PhD-level expertise across multiple disciplines
- Handles: Academic questions, research topics, conceptual explanations, interdisciplinary topics
- Strengths: Deep knowledge, analytical thinking, comprehensive explanations

BOB: Technical Implementation Agent specializing in software and technical solutions
- Handles: Programming, system design, technical implementation, code-related questions
- Strengths: Practical solutions, technical accuracy, implementation guidance

ROUTING GUIDELINES:
- Route to ALICE for: General knowledge, academic topics, research questions, conceptual explanations
- Route to BOB for: Technical implementation, programming, system design, code-related questions
- If both agents are suitable, prefer the agent with the lower workload (fewer tasks handled in this session)
- Alternate agents for ambiguous or general questions to ensure fair distribution
- Use FINISH when: All aspects of the user's question have been comprehensively addressed

DECISION FORMAT:
- To route: "ROUTE_TO: [AgentName] - [Brief, direct reason for routing]"
- To finish: "FINISH - [Comprehensive conclusion summarizing key points]"

IMPORTANT: Always route to a team member if the task requires further work. Only use FINISH when all aspects are fully addressed. Consider both expertise and workload/fairness in your decision.

Current Workload:
Alice: {alice_workload} tasks
Bob: {bob_workload} tasks

Conversation History:
{messages}"""),
    ("user", "{input}"),
])

# --- Enhanced Agent Creation ---

# Create tool-calling agents with better error handling
alice_tool_agent = create_tool_calling_agent(model_alice, web_tools, alice_tool_prompt)
alice_tool_executor = AgentExecutor(
    agent=alice_tool_agent, 
    tools=web_tools, 
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate"
)

bob_tool_agent = create_tool_calling_agent(model_bob, web_tools, bob_tool_prompt)
bob_tool_executor = AgentExecutor(
    agent=bob_tool_agent, 
    tools=web_tools, 
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate"
)

# Enhanced simple runnables with better streaming
alice_simple_runnable = alice_simple_prompt | model_alice | StrOutputParser()
bob_simple_runnable = bob_simple_prompt | model_bob | StrOutputParser()

# --- Enhanced Graph State ---

class AgentState(TypedDict):
    messages: Annotated[list[Union[HumanMessage, AIMessage]], operator.add]
    next_agent: str
    metadata: Dict[str, Any]  # For tracking conversation metadata

# --- Enhanced Agent Nodes ---

def agent_node(state: AgentState, agent_simple, agent_tools, name: str):
    """Enhanced agent node with better error handling and performance optimization."""
    current_question = state["messages"][-1].content
    
    # Enhanced difficulty assessment
    difficulty_analysis = is_difficult_question(current_question)
    use_tools = difficulty_analysis["use_tools"]
    
    logger.info(f"[{name}] Processing question: {current_question[:50]}...")
    logger.info(f"[{name}] Difficulty analysis: {difficulty_analysis}")
    
    full_content = ""
    start_time = time.time()
    
    try:
        if use_tools:
            logger.info(f"[{name}] Using tools for difficult question")
            
            # Use tool-enabled agent with better error handling
            chat_history = [msg for msg in state["messages"][:-1]]
            
            try:
                result = agent_tools.invoke({
                    "input": current_question,
                    "chat_history": chat_history
                })
                
                full_content = result["output"]
                logger.info(f"[{name}] Tool execution successful, response length: {len(full_content)}")
                
            except Exception as e:
                logger.error(f"[{name}] Tool execution failed: {str(e)}")
                # Enhanced fallback to simple agent
                logger.info(f"[{name}] Falling back to simple response")
                
                for chunk in agent_simple.stream({
                    "input": current_question,
                    "chat_history": state["messages"],
                }):
                    if chunk:
                        full_content += chunk
        else:
            logger.info(f"[{name}] Using simple mode")
            
            # Use simple agent with streaming
            for chunk in agent_simple.stream({
                "input": current_question,
                "chat_history": state["messages"],
            }):
                if chunk:
                    full_content += chunk
        
        processing_time = time.time() - start_time
        logger.info(f"[{name}] Response completed in {processing_time:.2f}s, length: {len(full_content)}")
        
        # Add metadata to state
        metadata = state.get("metadata", {})
        metadata[f"{name.lower()}_processing_time"] = processing_time
        metadata[f"{name.lower()}_used_tools"] = use_tools
        metadata[f"{name.lower()}_difficulty_score"] = difficulty_analysis["complexity_score"]
        
        return {
            "messages": [AIMessage(content=full_content, name=name)],
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"[{name}] Critical error: {str(e)}")
        error_message = f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question or ask for a different approach."
        return {
            "messages": [AIMessage(content=error_message, name=name)],
            "metadata": {"error": str(e)}
        }

# --- Enhanced Supervisor Node ---

members = ["Alice", "Bob"]

def supervisor_node(state: AgentState):
    """Enhanced supervisor node with dynamic, fair task distribution."""
    last_message = state["messages"][-1].content
    metadata = state.get("metadata", {})
    # Defensive: always provide integer values for workloads
    alice_workload = int(metadata.get("alice_workload", 0) or 0)
    bob_workload = int(metadata.get("bob_workload", 0) or 0)
    last_agent = metadata.get("last_agent", None)

    logger.info("[Supervisor] Making routing decision")
    start_time = time.time()

    try:
        supervisor_input = {
            "messages": state["messages"],
            "input": last_message,
            "members": ", ".join(members),
            "alice_workload": alice_workload,
            "bob_workload": bob_workload,
        }
        full_decision = ""
        for chunk in model_supervisor.stream(supervisor_routing_prompt.format_messages(**supervisor_input)):
            if hasattr(chunk, 'content') and chunk.content:
                full_decision += chunk.content

        supervisor_message = AIMessage(content=full_decision, name="Supervisor")
        next_agent = _parse_supervisor_decision(full_decision)

        # Workload balancing: if ambiguous, alternate or pick less-busy agent
        if next_agent not in members and next_agent != "FINISH":
            # Alternate if last_agent exists, else pick less-busy
            if last_agent == "Alice":
                next_agent = "Bob"
            elif last_agent == "Bob":
                next_agent = "Alice"
            else:
                next_agent = "Alice" if alice_workload <= bob_workload else "Bob"

        processing_time = time.time() - start_time
        logger.info(f"[Supervisor] Decision made in {processing_time:.2f}s: {next_agent}")

        # Update workload
        if next_agent == "Alice":
            alice_workload += 1
        elif next_agent == "Bob":
            bob_workload += 1
        if next_agent in members:
            metadata["last_agent"] = next_agent
        metadata["alice_workload"] = alice_workload
        metadata["bob_workload"] = bob_workload
        metadata["supervisor_processing_time"] = processing_time
        metadata["supervisor_decision"] = next_agent

        return {
            "messages": [supervisor_message],
            "next_agent": next_agent,
            "metadata": metadata
        }

    except Exception as e:
        logger.error(f"[Supervisor] Error in decision making: {str(e)}")
        # Default to alternate or less-busy agent
        if last_agent == "Alice":
            fallback_agent = "Bob"
        elif last_agent == "Bob":
            fallback_agent = "Alice"
        else:
            fallback_agent = "Alice" if alice_workload <= bob_workload else "Bob"
        return {
            "messages": [AIMessage(content=f"ROUTE_TO: {fallback_agent} - Default routing due to decision error", name="Supervisor")],
            "next_agent": fallback_agent,
            "metadata": {"error": str(e)}
        }

def _enhanced_supervisor_decision(state: AgentState):
    """Enhanced supervisor decision with better prompt engineering."""
    supervisor_input = {
        "messages": state["messages"],
        "input": state["messages"][-1].content,
        "members": ", ".join(members),
    }
    
    full_decision = ""
    for chunk in model_supervisor.stream(supervisor_routing_prompt.format_messages(**supervisor_input)):
        if hasattr(chunk, 'content') and chunk.content:
            full_decision += chunk.content
    
    return full_decision

def _parse_supervisor_decision(decision: str) -> str:
    """Enhanced decision parsing with better logic."""
    decision_lower = decision.lower()
    
    # Check for explicit routing
    if "route_to:" in decision_lower:
        for member in members:
            if member.lower() in decision_lower:
                return member
    
    # Check for finish
    if "finish" in decision_lower:
        return "FINISH"
    
    # Fallback: check for agent names in the decision
    for member in members:
        if member.lower() in decision_lower:
            return member
    
    # Default to Alice for general questions
    return "Alice"

# --- Enhanced Graph Construction ---

builder = StateGraph(AgentState)

# Add nodes with enhanced error handling
builder.add_node("Alice", lambda state: agent_node(state, alice_simple_runnable, alice_tool_executor, "Alice"))
builder.add_node("Bob", lambda state: agent_node(state, bob_simple_runnable, bob_tool_executor, "Bob"))
builder.add_node("Supervisor", supervisor_node)

# Add edges
for member in members:
    builder.add_edge(member, "Supervisor")

builder.add_conditional_edges(
    "Supervisor",
    lambda state: state.get("next_agent"),
    {"Alice": "Alice", "Bob": "Bob", "FINISH": END},
)

builder.set_entry_point("Supervisor")

app = builder.compile()

# --- Performance Monitoring ---

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0

performance_monitor = PerformanceMonitor()

# --- Enhanced Execution ---

if __name__ == "__main__":
    question = "Can you explain quantum computing in detail and provide a simple implementation example?"
    initial_input = HumanMessage(content=question)
    inputs = {"messages": [initial_input], "metadata": {}}
    
    num_printed_messages = 0
    
    print(f"User:\n{initial_input.content}")
    print("\n--- Starting Enhanced Multi-Agent Conversation ---")
    
    start_time = time.time()
    
    for state in app.stream(inputs, stream_mode="values"):
        current_messages = state['messages']
        
        if len(current_messages) > num_printed_messages:
            new_messages = current_messages[num_printed_messages:]
            
            for msg in new_messages:
                if isinstance(msg, HumanMessage):
                    print(f"\n{msg.name if msg.name else 'User'}:")
                    print(msg.content)
                elif isinstance(msg, AIMessage):
                    print(f"\n{msg.name}:")
                    print(msg.content)
            
            num_printed_messages = len(current_messages)
        
        if "next_agent" in state:
            if state["next_agent"] == "FINISH":
                print("\n--- Supervisor decided to FINISH the conversation ---")
                break
            elif state["next_agent"] in members:
                print(f"\n--- Supervisor routing to: {state['next_agent']} ---")
    
    total_time = time.time() - start_time
    print(f"\n--- Enhanced Conversation Complete in {total_time:.2f}s ---")
    
    # Print performance metrics
    if "metadata" in state:
        print(f"\nPerformance Metrics:")
        for key, value in state["metadata"].items():
            print(f"  {key}: {value}")