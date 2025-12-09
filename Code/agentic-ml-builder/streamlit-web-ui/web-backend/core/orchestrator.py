"""
File: orchestrator.py

Module that implements the MLOrchestrator class for coordinating and managing
the multi-agent workflow. This orchestrator routes user intents, manages
conversation state, and coordinates between various specialized agents
(spec agent, scaffold agent, conversation agent) and the template retriever
to build ML pipelines progressively.


"""

from core.spec_agent import SpecAgent
from core.retriever import TemplateRetriever
from core.scaffold_agent import ScaffoldAgent
from core.conversation_agent import ConversationAgent


class MLOrchestrator:
    """
    Orchestrates the multi-agent ML pipeline generation workflow.
    
    Coordinates between conversation agent, spec agent, scaffold agent,
    and template retriever to manage the complete cycle of requirements
    gathering, specification, code generation, and refinement.
    """
    def __init__(self):
        print("Initializing Agents...")
        self.spec_agent = SpecAgent()
        self.retriever = TemplateRetriever()
        self.scaffold_agent = ScaffoldAgent()
        self.conversation_agent = ConversationAgent()
        self.active_project = None # Stores {spec, templates, code}

    def route_intent(self, user_message: str) -> str:
        """
        Determines the user's intent based on message and state.
        Returns: "NEW", "REFINE", "ACTION_VALIDATE"
        """
        if not self.active_project:
            return "NEW"
        
        # Simple keyword heuristics (could be LLM based for robustness)
        msg = user_message.lower()
        if "start over" in msg or "new project" in msg or "create a" in msg:
            return "NEW"
        
        if "validate" in msg or "check code" in msg:
            return "ACTION_VALIDATE"
            
        # Default to refinement if we have an active project and it's not a new request
        return "REFINE"

    def handle_message(self, user_message: str, conversation_history: list):
        """
        Handles a single message in the conversation with context awareness.
        """
        intent = self.route_intent(user_message)
        
        # 1. Handle Refinement (Modify existing code)
        if intent == "REFINE":
            print(f"Refining active project with instruction: {user_message}")
            code = self.active_project["code"]
            
            # 1. RAG Retrieve (Find templates relevant to the NEW request)
            # e.g. "Add confusion matrix" -> finds plotting templates
            refine_templates = self.retriever.retrieve(user_message)
            print(f"RAG Refinement found {len(refine_templates)} templates.")
            
            # 2. Construct Rich Instruction
            rich_instruction = f"User Request: {user_message}\n\n"
            if refine_templates:
                rich_instruction += "Use these REFERENCE TEMPLATES if helpful:\n"
                for t in refine_templates:
                    rich_instruction += f"--- {t['filename']} ---\n{t['content'][:1000]}...\n\n"
            
            # 3. Apply changes via ValidatorAgent (acts as Refiner)
            new_code = self.fix_code(code, [], user_feedback=rich_instruction)
            
            # Update state
            self.active_project["code"] = new_code
            
            return f"I've updated the code using RAG-retrieved templates based on: **{user_message}**", self.active_project, True

        # 2. Handle Action (Run tools)
        if intent == "ACTION_VALIDATE":
             # We let the UI handle the button click usually, but here we can trigger a response
             return "I've ready to validate. Please click the 'Validate Code' button below to see the results.", self.active_project, True

        # 3. New Project (Standard Flow)
        # Clear active project if starting over explicitly? 
        # For now, just generate new.
        
        # Add current message to history for analysis
        full_history = conversation_history + [{"role": "user", "content": user_message}]
        
        # Check if we should ask questions or execute
        decision = self.conversation_agent.should_ask_questions(full_history)
        
        if not decision.get("ready", False):
            # Ask clarifying questions
            questions = decision.get("questions", [])
            if questions:
                response = "I need more information:\n\n" + "\n".join([f"- {q}" for q in questions])
            else:
                response = "Could you provide more details about your ML task?"
            return response, None, False
        
        # We have enough info - execute the pipeline
        try:
            # Summarize all conversation into a single query
            summary_prompt = f"Summarize this conversation into a concise ML task description:\n\n" + "\n".join([f"{m['role']}: {m['content']}" for m in full_history])
            
            spec = self.spec_agent.analyze(summary_prompt)
            if "error" in spec:
                return f"Error: {spec['error']}", None, True

            # Retrieve Templates
            meta = spec.get("data_set_description", {})
            tags = " ".join(meta.get("tag", []))
            desc = meta.get("description", "")
            search_query = f"{tags} {desc}"
            
            templates = self.retriever.retrieve(search_query)
            
            # Log retrieved templates
            print(f"RAG Retrieved {len(templates)} templates for query '{search_query[:50]}...':")
            for t in templates:
                print(f"  - {t['filename']}")

            # Generate Code
            code = self.scaffold_agent.generate(spec, templates)

            artifacts = {"spec": spec, "templates": templates, "code": code}
            self.active_project = artifacts # Set active project
            
            dataset_name = spec.get('data_set_description', {}).get('name', 'Unknown')
            response = f"Perfect! I've found the **{dataset_name}** dataset and generated your scaffold."
            
            return response, artifacts, True
            
        except Exception as e:
            return f"Error during execution: {str(e)}", None, True

    def validate_code(self, code: str):
        """Validates generated code using ValidatorAgent."""
        from core.validator_agent import ValidatorAgent
        if not hasattr(self, 'validator_agent'):
            self.validator_agent = ValidatorAgent()
        
        return self.validator_agent.validate_code(code)
    
    def fix_code(self, code: str, errors: list, user_feedback: str = None):
        """Fixes code using ValidatorAgent."""
        from core.validator_agent import ValidatorAgent
        if not hasattr(self, 'validator_agent'):
            self.validator_agent = ValidatorAgent()
            
        return self.validator_agent.fix_code(code, errors, user_feedback)

    def generate_from_json(self, spec_json: dict):
        """
        Generates code directly from a Spec JSON, bypassing the chat/search.
        """
        try:
            # 1. Extract Metadata for RAG
            meta = spec_json.get("data_set_description", {})
            tags = " ".join(meta.get("tag", []))
            desc = meta.get("description", "")
            search_query = f"{tags} {desc}"
            
            # 2. Retrieve Templates
            templates = self.retriever.retrieve(search_query)
            print(f"RAG Retrieved {len(templates)} templates for JSON spec...")
            
            # 3. Generate Code
            code = self.scaffold_agent.generate(spec_json, templates)
            
            # 4. Return Artifacts
            artifacts = {"spec": spec_json, "templates": templates, "code": code}
            self.active_project = artifacts # Set active project
            
            dataset_name = meta.get('name', 'Unknown')
            response = f"**Bypass Mode Active**: Generated code for uploaded spec: **{dataset_name}**"
            
            return response, artifacts, True
            
        except Exception as e:
            return f"Error during JSON generation: {str(e)}", None, True
