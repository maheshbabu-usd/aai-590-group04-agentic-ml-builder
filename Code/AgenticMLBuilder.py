import streamlit as st
import json
import os
from datetime import datetime
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless required explicitly 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(
    page_title="Agentic ML Builder",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_custom_css():
    st.markdown("""
    <style>
    /* Main background - light blue */
    .stApp {
        background-color: #E3F2FD;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #1976D2 0%, #2196F3 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    /* Left pane - chatbot area */
    [data-testid="column"]:first-child {
        background-color: #B3E5FC;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Right pane - JSON area */
    [data-testid="column"]:last-child {
        background-color: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subheaders */
    h3 {
        color: #1565C0;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Progress bar container */
    .stProgress {
        background-color: #C8E6C9;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50 0%, #81C784 100%);
    }
    
    /* Status bar - caption at bottom */
    .stCaption {
        background-color: #CFD8DC;
        padding: 10px;
        border-radius: 5px;
        font-weight: 500;
        color: #37474F;
        text-align: center;
    }
    
    /* Primary buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #388E3C 0%, #4CAF50 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Secondary buttons */
    .stButton button {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #F57C00 0%, #FF9800 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #2196F3 0%, #42A5F5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #1976D2 0%, #2196F3 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Text input fields */
    .stTextInput input {
        background-color: white;
        border: 2px solid #90CAF9;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stTextInput input:focus {
        border-color: #2196F3;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.2);
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: white;
        border: 2px solid #FFE0B2;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border-color: #FF9800;
        box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Divider */
    hr {
        border-color: #90CAF9;
        margin: 20px 0;
    }
    
    /* JSON display */
    .stJson {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        border: 2px solid #FFCC80;
    }
    
    /* Splitter slider styling */
    .stSlider {
        padding: 10px 0;
    }
    
    .stSlider > div > div > div {
        background-color: #90CAF9;
    }
    
    .stSlider > div > div > div > div {
        background-color: #1976D2;
    }
    
    /* Slider thumb */
    .stSlider [role="slider"] {
        background-color: #2196F3;
        border: 3px solid white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your ML Project Specification Assistant. I'll help you define your machine learning project by asking clarifying questions. Please describe your ML project idea to get started."
        })
    
    if 'project_spec' not in st.session_state:
        st.session_state.project_spec = {
            "project_name": "",
            "description": "",
            "ml_task_type": "",
            "data_sources": [],
            "features": [],
            "target_variable": "",
            "model_type": "",
            "evaluation_metrics": [],
            "deployment_requirements": {},
            "additional_notes": ""
        }
    
    if 'completion_percentage' not in st.session_state:
        st.session_state.completion_percentage = 0
    
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready to start"
    
    if 'json_editable' not in st.session_state:
        st.session_state.json_editable = False
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
    
    if 'column_ratio' not in st.session_state:
        st.session_state.column_ratio = 50

def calculate_completion():
    spec = st.session_state.project_spec
    total_fields = 9
    completed_fields = 0
    
    if spec.get('project_name'):
        completed_fields += 1
    if spec.get('description'):
        completed_fields += 1
    if spec.get('ml_task_type'):
        completed_fields += 1
    if spec.get('data_sources'):
        completed_fields += 1
    if spec.get('features'):
        completed_fields += 1
    if spec.get('target_variable'):
        completed_fields += 1
    if spec.get('model_type'):
        completed_fields += 1
    if spec.get('evaluation_metrics'):
        completed_fields += 1
    if spec.get('deployment_requirements'):
        completed_fields += 1
    
    return int((completed_fields / total_fields) * 100)

def extract_json_from_response(text):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except:
        pass
    return None

def chat_with_gpt(user_message):
    st.session_state.conversation_context.append({
        "role": "user",
        "content": user_message
    })
    
    system_prompt = f"""You are an expert ML project specification assistant. Your goal is to ask clarifying questions to gather complete information about the user's ML project.

Current project specification:
{json.dumps(st.session_state.project_spec, indent=2)}

Completion: {st.session_state.completion_percentage}%

Instructions:
1. Ask focused, specific questions to fill in missing information
2. When you have gathered enough information (95%+ complete), generate a complete JSON specification
3. Be conversational and helpful
4. Ask one or two questions at a time
5. If the user provides new information, acknowledge it and ask follow-up questions

If you determine the specification is complete enough, respond with:
"SPECIFICATION_COMPLETE: [your confirmation message]" followed by the complete JSON specification.

Otherwise, ask your next clarifying question(s)."""

    messages = [{"role": "system", "content": system_prompt}] + st.session_state.conversation_context[-10:]
    
    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        max_completion_tokens=2048
    )
    
    assistant_message = response.choices[0].message.content or ""
    
    st.session_state.conversation_context.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    if "SPECIFICATION_COMPLETE:" in assistant_message:
        extracted_json = extract_json_from_response(assistant_message)
        if extracted_json:
            st.session_state.project_spec.update(extracted_json)
            st.session_state.completion_percentage = 100
            st.session_state.status_message = "Specification complete!"
    else:
        spec_update_prompt = f"""Based on the conversation, extract any new information about the ML project and return ONLY a JSON object with the fields that should be updated. 

User said: {user_message}

Current spec: {json.dumps(st.session_state.project_spec)}

Return only valid JSON with fields to update, or an empty object {{}} if no updates needed."""
        
        try:
            spec_response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": spec_update_prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=1024
            )
            
            content = spec_response.choices[0].message.content or "{}"
            updates = json.loads(content)
            if updates:
                st.session_state.project_spec.update(updates)
                st.session_state.completion_percentage = calculate_completion()
                st.session_state.status_message = f"Updated specification ({st.session_state.completion_percentage}% complete)"
        except Exception as e:
            st.session_state.status_message = f"Chatting... ({st.session_state.completion_percentage}% complete)"
    
    return assistant_message

def main():
    initialize_session_state()
    apply_custom_css()
    
    st.title("Agentic ML Builder")
    
    splitter_col1, splitter_col2, splitter_col3 = st.columns([1, 8, 1])
    with splitter_col2:
        column_ratio = st.slider(
            "Adjust Pane Width",
            min_value=20,
            max_value=80,
            value=st.session_state.column_ratio,
            key="column_ratio_slider",
            label_visibility="collapsed",
            help="Drag to resize left and right panes"
        )
        if column_ratio != st.session_state.column_ratio:
            st.session_state.column_ratio = column_ratio
    
    left_width = st.session_state.column_ratio
    right_width = 100 - st.session_state.column_ratio
    
    left_col, right_col = st.columns([left_width, right_width])
    
    with left_col:
        st.subheader("Project Specification Conversation")
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        input_container = st.container()
        with input_container:
            col1, col2, col3 = st.columns([8, 1, 1])
            
            with col1:
                user_input = st.text_input(
                    "Type your message here...",
                    key="user_input",
                    label_visibility="collapsed",
                    placeholder="Describe your ML project or answer questions..."
                )
            
            with col2:
                voice_button = st.button("🎤", help="Voice input (browser-based)")
            
            with col3:
                send_button = st.button("↑", type="primary", help="Send message")
            
            if voice_button:
                st.info("Voice input requires browser microphone permissions. Please use the text input for now.")
            
            if send_button and user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                with st.spinner("Thinking..."):
                    response = chat_with_gpt(user_input)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.divider()
        
        st.progress(st.session_state.completion_percentage / 100, 
                   text=f"Specification Completion: {st.session_state.completion_percentage}%")
        
        st.caption(f"Status: {st.session_state.status_message}")
    
    with right_col:
        st.subheader("Project Specification (JSON)")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload", type=['json'], label_visibility="collapsed", key="upload")
            if uploaded_file:
                try:
                    uploaded_spec = json.load(uploaded_file)
                    st.session_state.project_spec = uploaded_spec
                    st.session_state.completion_percentage = calculate_completion()
                    st.session_state.status_message = "JSON file uploaded successfully"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")
        
        with col2:
            if st.button("Save", use_container_width=True):
                st.session_state.status_message = "Specification saved to session"
                st.success("Saved!")
        
        with col3:
            json_str = json.dumps(st.session_state.project_spec, indent=2)
            st.download_button(
                label="Download",
                data=json_str,
                file_name=f"ml_project_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        if st.button("✏️ Edit", use_container_width=True):
            st.session_state.json_editable = not st.session_state.json_editable
        
        if st.session_state.json_editable:
            edited_json = st.text_area(
                "Edit JSON",
                value=json.dumps(st.session_state.project_spec, indent=2),
                height=400,
                label_visibility="collapsed"
            )
            
            if st.button("Apply Changes", type="primary"):
                try:
                    st.session_state.project_spec = json.loads(edited_json)
                    st.session_state.completion_percentage = calculate_completion()
                    st.session_state.status_message = "JSON updated successfully"
                    st.session_state.json_editable = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
        else:
            st.json(st.session_state.project_spec)

if __name__ == "__main__":
    main()
