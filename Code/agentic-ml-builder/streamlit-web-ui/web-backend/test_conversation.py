
from core.conversation_agent import ConversationAgent
import json

agent = ConversationAgent()

print("--- Test 1: Vague Request ---")
history = [{"role": "user", "content": "I want to detect cars."}]
res = agent.should_ask_questions(history)
print(f"Decision: {res.get('ready')}")
print(f"Questions: {res.get('questions')}")

print("\n--- Test 2: Specific Request ---")
history = [{"role": "user", "content": "I want to build a PyTorch model to detect cars in images using object detection."}]
res = agent.should_ask_questions(history)
print(f"Decision: {res.get('ready')}")
print(f"Questions: {res.get('questions')}")

print("\n--- Test 3: Semi-Specific (Missing framework) ---")
history = [{"role": "user", "content": "Find me a dataset for sentiment analysis on twitter data."}]
res = agent.should_ask_questions(history)
print(f"Decision: {res.get('ready')}")
print(f"Questions: {res.get('questions')}")
