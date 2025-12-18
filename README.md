# **AAI-590 Final Capstone Project**

# Agentic ML Builder

This Capstone Project is a part of the AAI-590 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status:** Completed

**Project Objective:**

The main objective of this project is to Convert ML Project Specification involving weeks of engineering effort into fully functional ML Project Pipeline in minutes, using Agentic AI code generation guided by templates and best practices.
Our goal is to shrink that whole process from weeks to minutes. Imagine giving a simple project description and instantly getting a production-ready ML pipeline built on best-practice templates.

**Partner(s)/Contributor(s):**

  - Mahesh Babu Arcot Krishnamurthy
  - Ranjeet Das

**Project Description:**

Agentic ML Builder is a conversational, agent-orchestrated ML Builder that turns high-level requirements into enterprise-grade ML and MLOps pipelines at scale using Generative AI & Agentic AI, including templated context for code / document generation using RAG.

**Challenges:**
  - Too much time to convert ML Project Idea / Spec -> fully functional ML pipeline
  - No unified path from requirements to production-ready pipelines.
  - Manual and inconsistent ML model exploration, model setup / MLOps slows delivery.
  - Hinders experimentation, bottlenecks in onboarding, scalability. 

**Business Overview**
  - **Accelerated ML Engineering**
      The platform transforms natural-language project descriptions into ready-to-run ML pipelines, reducing lead times from days or weeks to minutes.
  - **Reduced Engineering Overhead**
      Automated specification, model selection, code generation, and validation dramatically reduce the operational burden on ML engineers.
  - **Production-Ready MLOps Integration**
      Generated scaffolds include tests, CI workflows, and deployment configurations suitable for enterprise environments including Azure ML and GitHub Actions.
  **ROI**
    - Automates repetitive ML setup tasks, saving 20 to 40 hours per project.
    - Ensures standardized, reproducible, and validated MLOps structures.
    - Bridges skill gaps between data scientists and DevOps engineers.
    - Seamlessly integrates with Azure AI Foundry and GitHub Actions.

**Methods Used:**
  - Code and Document Generation using Agents
  - Context from Code Templates and Best Practices
  - Multi-Agent Orchestration and Communication
  - Natural Conversational intake
  - MLOps Automation
  - Developer Productivity & CI/CD

**Technologies:**
  - Streamlit conversational intake
  - MCP for multi-agent communication and orchestration
  - Agentic AI and Agents for Intelligence and Generation
  - OpenAI SDK, Azure AI Inference, LangChain, RAG: ChromaDB, FAISS, Sentence Transformers
  - PyTorch, TorchVision, Scikit-Learn, Pandas, NumPy, TensorBoard
  - Azure AI Foundry + Azure ML orchestration
  - OneDrive export via MSAL / MS Graph SDK
  - Pytest, Docker, Flake8, Click, Rich, YAML, Jinja2



  **Frameworks:** PyTorch, TensorFlow, Scikit-Learn

  **Libraries:** / Model: PyTorch CNN Architectures (Vision Models)
                          Transformer Based Models (NLP Tasks, Sentiment Analysis)
                          XGBoost Implementations (Clustering Algorithms, Ensemble methods)

  **Tools:** NVIDIA CUDA for GPU acceleration

  **IDE:**  VS Code

  **Hardware:** GPUs (e.g., NVIDIA RTX 30xx series, NVIDIA A100, NVIDIA L4, Tesla T4) for real-time processing and training large models.

**Dataset:**
    
Sample Image, Audio, Text and Video datasets from OpenML, Kaggle

**Installation and Execution**

This project can be use from the Web UI and well as VS Code / Powershell from local windows system

To set up this project

1. **Clone the Repository:**
   Open a terminal and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/maheshbabu-usd/aai-590-group04-agentic-ml-builder/tree/main/Code/agentic-ml-builder
   ```

2. **Navigate to the Project Directory:**
   Change into the project directory:
   ```bash
   cd agentic-ml-builder
   ```

3. **To setup and run the project**
   Follow the instruction in the README.md in the root folder
   For a successful run and project code generation
     - Setting up the OpenAI Key is mandatory for using gpt-4o for code generation
     - Azure OpenAI and Agentic AI (Optional) if running locally
     - Ensure that the input and output folders created and available locally
     - Ensure the input project scope specification is available in the input folder
     - Follow the instruction in the 01.README.md in the root folder
   
5. **Install Dependencies:**
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

**Acknowledgments:**

We would like to thank **Prof. Dr. Zahid Wani** for his invaluable guidance and mentorship throughout the course. 

**References:**

[1] Shi, Y., Wang, M., Cao, Y., Lai, H., Lan, J., Han, X., Wang, Y., Geng, J., Li, Z., Xia, Z., Chen, X., Li, C., Xu, J., Duan, W., & Zhu, Y. (2025). Aime: Towards fully-autonomous multi-agent framework (arXiv:2507.11988). arXiv. https://doi.org/10.48550/arXiv.2507.11988 

[2] Higuchi, T., Henry, S., & Straight, E. (2025, October 1). Introducing Microsoft Agent Framework: The open-source engine for agentic AI apps. Azure AI Foundry Blog. https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/

[3] Microsoft. (2025). agent-framework: A framework for building, orchestrating and deploying AI agents and multi-agent workflows [Computer software]. GitHub. https://github.com/microsoft/agent-framework

[4] Ashrafi, N., Bouktif, S., & Mediani, M. (2025). Enhancing LLM code generation: A systematic evaluation of multi-agent collaboration and runtime debugging for improved accuracy, reliability, and latency (arXiv preprint arXiv:2505.02133 v1). arXiv. https://doi.org/10.48550/arXiv.2505.02133)

[5] Eken, B., Pallewatta, S., Tran, N. K., Tosun, A., & Babar, M. A. (2024). *A multivocal review of MLOps practices, challenges and open issues* (arXiv preprint arXiv:2406.09737v2). https://arxiv.org/abs/2406.09737

[6] OpenAI. (2024). A practical guide to building agents [PDF]. https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf


