"""
================================================================================
Agentic ML Builder - Streamlit Web UI Main Application
================================================================================

Module:         app.py
Description:    Main entry point for the Agentic ML Builder web interface.
                This Streamlit application provides a user-friendly chat-based
                interface for designing and building ML pipelines. Users can
                interact with AI agents to find datasets, generate code, validate
                solutions, and run Python scripts in a sandboxed environment.


Key Features:
    - Natural language interaction with ML agents
    - Direct JSON spec input for ML pipeline configuration
    - Real-time code validation and auto-fixing capabilities
    - Integrated playground IDE with code execution
    - Dataset search and retrieval
    - Template-based code generation

Dependencies:
    - streamlit: Web UI framework
    - core.orchestrator: ML orchestration engine
    - dotenv: Environment variable management
    - base64: Image encoding
    - json: JSON file handling
    - subprocess: Code execution
    - sys: System utilities

================================================================================
"""

import streamlit as st
import os
from core.orchestrator import MLOrchestrator
from dotenv import load_dotenv

from dotenv import load_dotenv
import json
import subprocess
import sys

load_dotenv()

import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config(page_title="Agentic ML Builder", page_icon="assets/icon.png", layout="wide")

# Custom Title with Icon (Base64 to remove expand button)
col_logo, col_title = st.columns([1, 8])
with col_logo:
    img_base64 = get_base64_image("assets/icon.png")
    st.markdown(f'<img src="data:image/png;base64,{img_base64}" width="70" style="margin-top: 10px;">', unsafe_allow_html=True)
with col_title:
    st.title("Agentic ML Builder")

st.markdown("Chat with the agent to design your ML pipeline. I will find real datasets and generate code for you.")

# Initialize Orchestrator globally
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MLOrchestrator()

# Sidebar for Config
with st.sidebar:
    st.header("⚙️ Configuration") # Modified header
    if st.button("Clear History"): # Moved Clear History button
        st.session_state.messages = []
        st.rerun()

    if not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    st.info("System Components:\n- **SpecAgent**: NLU + Web Search\n- **RAG**: Template Retrieval\n- **ScaffoldAgent**: Code Generation\n - **Customized Templates**")
    
    st.divider()
    st.subheader("📥 Direct Spec Input")
    st.markdown(
    "<h3 style='font-size:20px; margin-bottom:0;'>Input Method</h3>",
    unsafe_allow_html=True
    )

    input_method = st.radio(
        "Input Method (hidden)",   # required label
        ["Upload File", "Paste JSON"],
        horizontal=True,
        label_visibility="collapsed"  # hides it properly
    )
    
    spec_json = None
    uploaded_file = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload spec.json", type=['json'])
        if uploaded_file:
            try:
                spec_json = json.load(uploaded_file)
                st.success(f"Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
    else:
        json_text = st.text_area("Paste JSON here", height=200)
        if json_text:
            try:
                spec_json = json.loads(json_text)
                st.success("JSON parsed successfully!")
            except Exception as e:
                if len(json_text) > 5: # Only show error if they started typing
                    st.error(f"Invalid JSON: {e}")

# Process Input (Bypass Mode)
if spec_json is not None:
    try:
        source_name = uploaded_file.name if uploaded_file else "Pasted JSON"
        st.info(f"📂 **Using Source**: {source_name}")
        
        if st.button("🚀 Generate from JSON"):
            with st.spinner("Generating code from uploaded spec..."):
                # Use the orchestrator from session state
                response_text, artifacts, is_final = st.session_state.orchestrator.generate_from_json(spec_json)
                
                st.markdown(response_text)
                
                if artifacts:
                    spec = artifacts["spec"]
                    templates = artifacts["templates"]
                    code = artifacts["code"]
                    
                    # --- Validation Section (Shared) ---
                    st.divider()
                    st.subheader("🕵️ Code Verification")
                    
                    col1, col2 = st.columns(2)
                    if col1.button("✅ Validate Code", key="validate_uploaded"): # Added key for uniqueness
                        with st.spinner("Running smoke tests and linting..."): # Added linting
                            res = st.session_state.orchestrator.validate_code(code)
                            if res['valid']: 
                                st.success(f"**Syntax Check Passed!** (Score: {res['score']}/10)") # Updated message
                            else: 
                                st.error(f"**Syntax Errors Found:**") # Updated message
                                for err in res['errors']:
                                    st.code(err)
                            
                            if res.get('feedback'):
                                st.info(f"**Reviewer Feedback:**\n{res['feedback']}")
                            
                            # Save result to session state to enable "Fix" button
                            st.session_state.last_validation = res
                            st.session_state.last_code = code
                    
                    if col2.button("🔧 Auto-Fix Logic", key="fix_uploaded"): # Added key for uniqueness
                        user_feedback = st.text_input("Custom Fix Instructions (Optional)", key="feedback_uploaded")
                        
                        if 'last_validation' in st.session_state and (not st.session_state.last_validation['valid'] or user_feedback):
                             with st.spinner("Attempting to fix code..."):
                                fixed_code = st.session_state.orchestrator.fix_code(
                                    st.session_state.last_code, 
                                    st.session_state.last_validation['errors'],
                                    user_feedback
                                )
                                st.markdown("### 🛠️ Fixed Code:")
                                st.code(fixed_code, language='python')
                        else:
                             st.warning("Run validation first, or provide instructions!")

                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Spec & Data", "🗂️ Templates", "💻 Code", "▶️ Playground"]) # Added Playground tab
                    with tab1: st.json(spec)
                    with tab2: 
                        if templates:
                            for t in templates:
                                with st.expander(f"📄 {t['filename']}"):
                                    st.code(t['content'], language='python')
                        else:
                            st.info("No templates used.")
                    with tab3: 
                        st.code(code, language='python')
                        st._button("📥 Download main.py", code, file_name="main.py", key="dl_uploaded")
                    
                    with tab4:
                        st.subheader("🛠️ Playground IDE")
                        
                        col_editor, col_terminal = st.columns([1.5, 1])
                        
                        with col_editor:
                            st.markdown("### 📝 Code Editor")
                            # Initialize editor state if fresh generation
                            if "editor_code_uploaded" not in st.session_state or st.session_state.get("last_generated_code") != code:
                                st.session_state.editor_code_uploaded = code
                                st.session_state.last_generated_code = code

                            # Editor Text Area
                            edited_code = st.text_area("Main Script", height=500, key="editor_code_uploaded", label_visibility="collapsed")
                            st.session_state.last_code = edited_code # Sync validatable code

                        with col_terminal:
                            st.markdown("### 🤖 Code Helper")
                            helper_prompt = st.text_input("Ask for changes (e.g. 'Add a progress bar')", key="helper_uploaded")
                            if st.button("✨ Apply Changes", key="apply_helper_uploaded"):
                                if helper_prompt:
                                    with st.spinner("AI is modifying your code..."):
                                        new_code = st.session_state.orchestrator.fix_code(edited_code, [], user_feedback=helper_prompt)
                                        st.session_state.editor_code_uploaded = new_code
                                        st.session_state.last_code = new_code
                                        st.rerun()

                            st.divider()
                            st.markdown("### 📟 Terminal Output")
                            
                            if st.button("▶️ Run Code", key="run_uploaded", type="primary"):
                                with st.spinner("Executing..."):
                                    try:
                                        with open("temp_run.py", "w") as f:
                                            f.write(edited_code)
                                        
                                        result = subprocess.run([sys.executable, "temp_run.py"], capture_output=True, text=True, timeout=30)
                                        
                                        # Terminal Styling
                                        terminal_bg = """
                                        <style>
                                        .terminal {
                                            background-color: #0e1117;
                                            color: #00ff00;
                                            font-family: 'Courier New', Courier, monospace;
                                            padding: 10px;
                                            border-radius: 5px;
                                            border: 1px solid #333;
                                            max-height: 300px;
                                            overflow-y: auto;
                                        }
                                        </style>
                                        """
                                        st.markdown(terminal_bg, unsafe_allow_html=True)
                                        
                                        output_content = result.stdout if result.stdout else (result.stderr if result.stderr else "Execution success (No Output).")
                                        color = "#00ff00" if not result.stderr or (result.stdout and not result.returncode) else "#ff4b4b"
                                        
                                        st.markdown(f'<div class="terminal" style="color: {color};">{output_content.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                                        if result.stderr:
                                            st.session_state.last_error = result.stderr
                                            
                                    except subprocess.TimeoutExpired:
                                        st.error("Time Limit Exceeded (30s)")
                                    except Exception as e:
                                        st.error(f"System Error: {e}")

                            # Runtime Fix
                            if 'last_error' in st.session_state and st.session_state.last_error:
                                 if st.button("🚑 Fix Runtime Error", key="fix_runtime_uploaded"):
                                    with st.spinner("Debugging..."):
                                        fixed_code = st.session_state.orchestrator.fix_code(
                                            edited_code, 
                                            ["Runtime Error"], 
                                            user_feedback=f"Fix this runtime error:\n{st.session_state.last_error}"
                                        )
                                        st.session_state.editor_code_uploaded = fixed_code
                                        st.session_state.last_error = None # Clear error
                                        st.rerun()
        
        # Stop here if file is present so we don't show the chat
        st.warning("⚠️ Chat disabled while using Direct Upload mode. Remove file to chat.")
        st.stop()
        
    except Exception as e:
        st.error(f"Invalid JSON file: {e}")

# Initialize Chat History (Only if no file uploaded)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello I am your ML Code Pilot. Describe the ML Project you want to build (e.g., 'Find a dataset for bird audio classification and build a PyTorch model')."}]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message contains complex artifacts (saved in specific keys), display them
        if "artifacts" in message:
            artifacts = message["artifacts"]
            spec = artifacts.get("spec")
            templates = artifacts.get("templates")
            code = artifacts.get("code")
            
            tab1, tab2, tab3 = st.tabs(["📋 Spec & Data", "🗂️ Templates", "💻 Code"])
            with tab1:
                st.json(spec)
            with tab2:
                for t in templates:
                    with st.expander(f"📄 {t['filename']}"):
                        st.code(t['content'], language='python')
            with tab3:
                st.code(code, language='python')
                st.download_button("📥 Download main.py", code, file_name="main.py",key=f"dl_{message.get('id', 'new')}")
# ---------- Chat Input ----------
prompt = st.chat_input("Describe your project...", key="main_chat_input")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Logic to handle processing
prompt = None
# We rely on submit_custom_chat to append to messages
# So we check if the last message is new

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    # Process with Orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                orchestrator = MLOrchestrator()
                
                # Use conversational handler
                response_text, artifacts, is_final = orchestrator.handle_message(
                    prompt, 
                    st.session_state.messages[:-1]  # All messages except the one we just added
                )
                
                st.markdown(response_text)
                
                # If artifacts are returned, this is the final response
                if artifacts:
                    spec = artifacts["spec"]
                    templates = artifacts["templates"]
                    code = artifacts["code"]
                    
                    source = spec.get('data_set_description', {}).get('source', 'unknown')
                    st.info(f"📊 **Dataset Source**: {source.upper()}")
                    
                    # Show warning if dataset search used fallback
                    if 'warning' in spec.get('data_set_description', {}):
                        st.warning(spec['data_set_description']['warning'])
                    
                    # --- Validation Section ---
                    st.divider()
                    st.subheader("🕵️ Code Verification")
                    
                    col1, col2 = st.columns(2)
                    
                    if col1.button("✅ Validate Code", key=f"validate_{len(st.session_state.messages)}"):
                        with st.spinner("Running smoke tests and linting..."):
                            validation_result = st.session_state.orchestrator.validate_code(code)
                            
                            if validation_result['valid']:
                                st.success(f"**Syntax Check Passed!** (Score: {validation_result['score']}/10)")
                            else:
                                st.error(f"**Syntax Errors Found:**")
                                for err in validation_result['errors']:
                                    st.code(err)
                            
                            if validation_result.get('feedback'):
                                st.info(f"**Reviewer Feedback:**\n{validation_result['feedback']}")
                            
                            # Save result to session state to enable "Fix" button
                            st.session_state.last_validation = validation_result
                            st.session_state.last_code = code

                    if col2.button("🔧 Auto-Fix Logic", key=f"fix_{len(st.session_state.messages)}"):
                        user_feedback = st.text_input("Custom Fix Instructions (Optional)", key=f"feedback_{len(st.session_state.messages)}")
                        
                        if 'last_validation' in st.session_state and (not st.session_state.last_validation['valid'] or user_feedback):
                             with st.spinner("Attempting to fix code..."):
                                fixed_code = st.session_state.orchestrator.fix_code(
                                    st.session_state.last_code, 
                                    st.session_state.last_validation['errors'],
                                    user_feedback
                                )
                                st.markdown("### 🛠️ Fixed Code:")
                                st.code(fixed_code, language='python')
                        else:
                             st.warning("Run validation first, or provide instructions!")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Spec & Data", "🗂️ Templates", "💻 Code", "▶️ Playground"])
                    with tab1:
                        st.json(spec)
                    with tab2:
                        if templates:
                            for t in templates:
                                with st.expander(f"📄 {t['filename']}"):
                                    st.code(t['content'], language='python')
                        else:
                            st.info("No templates used.")
                    with tab3:
                        st.code(code, language='python')
                        st.download_button("📥 Download main.py", code, file_name="main.py")

                    with tab4:
                        st.subheader("🛠️ Playground IDE")
                        run_key = f"run_{len(st.session_state.messages)}"
                        
                        col_editor, col_terminal = st.columns([1.5, 1])
                        
                        # Generate unique keys for widgets based on message index
                        editor_key = f"editor_code_{len(st.session_state.messages)}"
                            
                        with col_editor:
                            st.markdown("### 📝 Code Editor")
                            # Init editor
                            if editor_key not in st.session_state:
                                st.session_state[editor_key] = code
                            
                            edited_code = st.text_area("Main Script", height=500, key=editor_key, label_visibility="collapsed")
                            
                        with col_terminal:
                            st.markdown("### 🤖 Code Helper")
                            helper_key = f"helper_{len(st.session_state.messages)}"
                            helper_prompt = st.text_input("Ask for changes", key=helper_key)
                            
                            apply_key = f"apply_{len(st.session_state.messages)}"
                            if st.button("✨ Apply Changes", key=apply_key):
                                if helper_prompt:
                                    with st.spinner("AI is modifying code..."):
                                        new_code = st.session_state.orchestrator.fix_code(edited_code, [], user_feedback=helper_prompt)
                                        st.session_state[editor_key] = new_code
                                        st.rerun()

                            st.divider()
                            st.markdown("### 📟 Terminal Output")
                            
                            if st.button("▶️ Run Code", key=run_key, type="primary"):
                                with st.spinner("Executing..."):
                                    try:
                                        with open("temp_run.py", "w") as f:
                                            f.write(edited_code)
                                        
                                        result = subprocess.run([sys.executable, "temp_run.py"], capture_output=True, text=True, timeout=30)
                                        
                                        # Terminal Styling (Inline CSS)
                                        terminal_bg = """
                                        <style>
                                        .terminal {
                                            background-color: #0e1117;
                                            color: #00ff00;
                                            font-family: 'Courier New', Courier, monospace;
                                            padding: 10px;
                                            border-radius: 5px;
                                            border: 1px solid #333;
                                            max-height: 300px;
                                            overflow-y: auto;
                                        }
                                        </style>
                                        """
                                        st.markdown(terminal_bg, unsafe_allow_html=True)
                                        
                                        output_content = result.stdout if result.stdout else (result.stderr if result.stderr else "Execution success (No Output).")
                                        color = "#00ff00" if not result.stderr or (result.stdout and not result.returncode) else "#ff4b4b"
                                        
                                        st.markdown(f'<div class="terminal" style="color: {color};">{output_content.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                                        
                                        if result.stderr:
                                            st.session_state[f"error_{run_key}"] = result.stderr
                                            
                                    except Exception as e:
                                        st.error(f"Execution failed: {e}")

                            # Fix Runtime Error
                            error_key = f"error_{run_key}"
                            if error_key in st.session_state and st.session_state[error_key]:
                                 if st.button("🚑 Fix Runtime Error", key=f"fix_rt_{run_key}"):
                                    with st.spinner("Fixing logic..."):
                                        fixed_code = st.session_state.orchestrator.fix_code(
                                            edited_code, 
                                            ["Runtime Error"], 
                                            user_feedback=f"Fix this runtime error:\n{st.session_state[error_key]}"
                                        )
                                        st.session_state[editor_key] = fixed_code
                                        st.session_state[error_key] = None
                                        st.rerun()
                    
                    # Save with artifacts
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "artifacts": artifacts
                    })
                else:
                    # Just a question - save as regular message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text
                    })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
