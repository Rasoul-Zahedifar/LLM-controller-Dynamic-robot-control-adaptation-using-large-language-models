"""
LLM Chain Module

This module provides the LLM chain interface for controller gain tuning
and performance evaluation using language models.
"""

import re
from langchain_core.runnables.history import RunnableWithMessageHistory


class LLMChain:
    """
    LLM Chain for controller parameter tuning and performance evaluation.
    
    This class manages interactions with language models to:
    1. Suggest new controller gains based on performance
    2. Evaluate whether current performance is satisfactory
    
    Attributes:
        chain_with_message_history: Runnable chain with conversation history
        pattern_helper: Regex pattern for extracting controller gains
        pattern_satisfier: Regex pattern for extracting satisfaction status
        store: Dictionary for storing session data
    """
    
    def __init__(self, llm, prompt, chat_history_for_chain):
        """
        Initialize the LLM chain.
        
        Args:
            llm: Language model instance (e.g., ChatOpenAI)
            prompt: Prompt template for the conversation
            chat_history_for_chain: Chat history manager instance
        """
        # Create the chain by piping prompt to LLM
        chain = prompt | llm
        
        # Initialize storage
        self.store = {}
        
        # Create chain with message history for context-aware responses
        self.chain_with_message_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: chat_history_for_chain,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer"
        )
        
        # Regex patterns for parsing LLM responses
        self.pattern_helper = (
            r'new_controller_gains: \[k_p=\s*(.*?),\s*k_d=\s*(.*?),\s*k_i=\s*(.*?),\s*'
            r'landa_1=\s*(.*?),\s*landa_2=\s*(.*?)\]'
        )
        self.pattern_satisfier = r'is_performance_satisfactory:\s*(.*)'

    def is_float(self, value):
        """
        Check if a value can be converted to float.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if value can be converted to float, False otherwise
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def is_boolean(self, value):
        """
        Check if a value can be converted to boolean.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if value can be converted to boolean, False otherwise
        """
        try:
            bool(value)
            return True
        except (ValueError, TypeError):
            return False

    def run_helper(self, input_text):
        """
        Run the LLM chain to get new controller gains.
        
        This method repeatedly queries the LLM until it receives a valid
        response with all five controller gain parameters.
        
        Args:
            input_text: Input message describing current system state and performance
            
        Returns:
            Tuple of (k_p, k_d, k_i, landa_1, landa_2, full_response_text)
            where the first five are float values and the last is the complete
            LLM response text
        """
        while True:
            # Invoke the chain with the input
            output = self.chain_with_message_history.invoke(
                {"input": input_text},
                config={"configurable": {"session_id": "abc123"}}
            )
            
            # Try to extract controller gains from response
            matches = re.search(self.pattern_helper, output.content)
            
            if matches:
                # Extract the five gain parameters
                k_p = matches.group(1).strip()
                k_d = matches.group(2).strip()
                k_i = matches.group(3).strip()
                landa_1 = matches.group(4).strip()
                landa_2 = matches.group(5).strip()
                
                # Validate that all values are floats
                if all(self.is_float(value) for value in [k_p, k_d, k_i, landa_1, landa_2]):
                    return (
                        float(k_p),
                        float(k_d),
                        float(k_i),
                        float(landa_1),
                        float(landa_2),
                        output.content
                    )

    def run_satisfier(self, input_text):
        """
        Run the LLM chain to evaluate performance satisfaction.
        
        This method queries the LLM to determine if the current controller
        performance is satisfactory.
        
        Args:
            input_text: Input message describing current performance metrics
            
        Returns:
            Tuple of (is_satisfied, full_response_text)
            where is_satisfied is a boolean and full_response_text is the
            complete LLM response
        """
        while True:
            # Invoke the chain with the input
            output = self.chain_with_message_history.invoke(
                {"input": input_text},
                config={"configurable": {"session_id": "abc123"}}
            )
            
            # Try to extract satisfaction status from response
            matches = re.search(self.pattern_satisfier, output.content)
            
            if matches:
                performance_satisfactory = matches.group(1).strip()
                
                # Validate and convert to boolean
                if self.is_boolean(performance_satisfactory):
                    return eval(performance_satisfactory), output.content

