RAG_ChatPDF: AI-Powered Document Intelligence

🛠️ Tech Stacks
Python 3.9+, FastAPI, Uvicorn, LangChain, MongoDB / Frontend Node.js, NPM, React/Next.js

1. Backend Setup (Python & FastAPI)
Ensure you have Python 3.9+ installed. Open your terminal in the project root:

Bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Make sure you have your LLM
ollama pull llama3.1

# Start the FastAPI server
uvicorn main:app --reload --port 8000

2. Frontend Setup (Node.js & NPM)
Open a second terminal window in the project root:

Bash
# Install packages
npm install

# Launch the development server
npm run dev
