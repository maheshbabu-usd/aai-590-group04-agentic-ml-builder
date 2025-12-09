"""
File: conversation_agent.py

Module that implements the ConversationAgent class for multi-turn conversational
interactions. This agent manages the gathering of complete requirements through
an interactive dialogue with users before proceeding with ML pipeline generation.


"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()


class ConversationAgent:
    """Handles multi-turn conversation to gather complete requirements."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def should_ask_questions(self, conversation_history: list) -> dict:
        """
        Analyzes conversation and decides if ready to execute.
        Now more permissive - executes if there's ANY reasonable task description.
        """
        # Build conversation context
        conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
        
        system_prompt = """
        You are a Senior ML Engineer and Requirements Analyst.
        
        Your goal is to ensure we have a COMPLETE specification before generating any code.
        Do NOT let the user proceed with vague requests.
        
        REQUIRED INFORMATION (The "3 Pillars"):
        1. Task Type: What is the ML task? (e.g., Classification, Regression, Clustering, Object Detection)
        2. Data Modality: What kind of data? (e.g., Tabular/CSV, Images, Text, Audio)
        3. Framework Preference: What tools? (e.g., PyTorch, Sklearn, HuggingFace, XGBoost)
        
        ANALYSIS RULES:
        - If ANY of the 3 pillars are missing or unclear, set "ready": false.
        - Generate specific questions to gather the missing info.
        - If the user provides a dataset name (e.g. "iris"), infer the Modality (Tabular) but still ask for Framework if unknown.
        - If the user explicitly says "you choose" or "default", accept that as valid.
        
        Return JSON:
        {
          "ready": true/false,
          "questions": ["specific question 1", "specific question 2"],
          "reasoning": "Missing framework preference and data modality."
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Conversation History:\n{conv_text}\n\nDECISION:"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in ConversationAgent: {e}")
            # Fallback for safety, but try to ask clarification if possible
            return {"ready": True, "questions": [], "reasoning": "Error in agent"}
