"""
LLM Configuration Module

This module provides utilities for configuring and initializing language models
and prompts for the controller tuning system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables from .env file
# Look for .env in the project root (parent of llm_controller package)
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def get_llm_instance(api_key=None, model='gpt-3.5-turbo', temperature=0.7):
    """
    Create and return an LLM instance.
    
    Args:
        api_key: OpenAI API key (if None, reads from environment variable)
        model: Model name (default: 'gpt-3.5-turbo')
        temperature: Temperature parameter for generation (default: 0.7)
        
    Returns:
        ChatOpenAI instance
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError(
                "OpenAI API key not provided. Either pass it as an argument or "
                "set the OPENAI_API_KEY environment variable."
            )
    
    return ChatOpenAI(
        openai_api_key=api_key,
        model=model,
        temperature=temperature
    )


def create_prompt(system_message=None):
    """
    Create a prompt template for the LLM chain.
    
    Args:
        system_message: Optional custom system message. If None, uses default.
        
    Returns:
        ChatPromptTemplate instance
    """
    if system_message is None:
        system_message = """You are an expert control systems engineer specializing in 
robotic manipulator control. Your task is to help tune PID and adaptive sliding mode 
controller parameters based on system performance metrics.

When asked to suggest new controller gains, you should respond in the following format:
new_controller_gains: [k_p=<value>, k_d=<value>, k_i=<value>, landa_1=<value>, landa_2=<value>]

When asked to evaluate if performance is satisfactory, respond in the following format:
is_performance_satisfactory: <True or False>

Consider the following guidelines:
- k_p (proportional gain): Controls steady-state error, typically 0-100
- k_d (derivative gain): Controls damping, typically 0-80% of k_p
- k_i (integral gain): Eliminates steady-state error, typically 0-60% of k_p
- landa_1, landa_2 (adaptive gains): Control adaptation rate, typically 0-1

Higher gains lead to faster response but may cause oscillations or instability.
Lower gains lead to slower, more stable response."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    return prompt


def create_chat_history():
    """
    Create a chat message history instance.
    
    Returns:
        ChatMessageHistory instance
    """
    return ChatMessageHistory()

