
from core.validator_agent import ValidatorAgent

agent = ValidatorAgent()

print("Test 1: Valid Code")
valid_code = "import os\ndef hello():\n    print('Hello')"
res = agent.validate_code(valid_code)
print(f"Valid: {res['valid']}, Score: {res['score']}")

print("\nTest 2: Syntax Error")
invalid_code = "def hello()\n    print('Missing colon')"
res = agent.validate_code(invalid_code)
print(f"Valid: {res['valid']}, Errors: {res['errors']}")

print("\nTest 3: Quality Check (No Import)")
poor_code = "def foo(): pass"
res = agent.validate_code(poor_code)
print(f"Valid: {res['valid']}, Score: {res['score']}, Errors: {res['errors']}")
