# Agentic ML Architect - Local Setup Guide

This guide provides step-by-step instructions to set up and run the Agentic ML Architect application on your local Windows machine.

## Prerequisites
- Python 3.8 or higher installed.
- An OpenAI API Key (for GPT-4o access).

## Setup Instructions

1. **Navigate to the Project Directory**
   Open your terminal (Command Prompt or PowerShell) and navigate to the project folder.

2. **Create a Virtual Environment**
   It is recommended to use a virtual environment to manage dependencies. Run the following command to create one named `myenv`:
   
   python -m venv myenv

3. **Activate the Virtual Environment**
   Activate the environment to ensure libraries are installed in the isolated scope:
   
   myenv\Scripts\activate

   (You should see `(myenv)` appear at the start of your command prompt).

4. **Install Dependencies**
   Install all required Python libraries listed in `requirements.txt`:
   
   pip install -r requirements.txt

5. **Configure Environment Variables**
   The application requires an OpenAI API key.
   - Create a file named `.env` in the root directory (if it doesn't exist).
   - Add your API key to it:
     OPENAI_API_KEY=your_api_key_here

6. **Run the Application**
   Launch the Streamlit interface:
   
   streamlit run app.py

   The application should automatically open in your default web browser.

## Troubleshooting
- If `streamlit` is not recognized, ensure your virtual environment is activated.
- If you encounter "ModuleNotFoundError", try running `pip install -r requirements.txt` again.
