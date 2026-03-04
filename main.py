from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
import os
import shutil
import uvicorn

# Import your provided logic
from src.ingest import main as ingest_run
from src.search import search_similarity
from config import settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/ingest")
async def api_ingest(
    file: UploadFile = File(...), 
    strategy: str = Form(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50)
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Trigger ingest.py - Ensure your main() returns the list of chunks
        chunks = ingest_run(temp_path, file.filename, strategy, chunk_size, overlap)
        
        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "strategy": strategy,
            "count": len(chunks) if chunks else 0,
            "preview": chunks[:10] if chunks else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.get("/api/search")
async def api_search(query: str):
    try:
        # Fetch a few extra candidates just in case there are duplicates
        raw_results = search_similarity(query, top_k=6)
        
        # Deduplicate based on exact text
        unique_results = []
        seen_texts = set()
        
        for res in raw_results:
            text_snippet = res.get("text", "")
            if text_snippet not in seen_texts:
                unique_results.append(res)
                seen_texts.add(text_snippet)
            
            # Stop once we have exactly 3 UNIQUE chunks
            if len(unique_results) == 3:
                break
                
        return {"results": unique_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def api_generate(
    query: str = Form(...), 
    context: str = Form(...),
    conversation_history: str = Form("") # 1. Added this variable with a blank default
):
    try:
        llm = OllamaLLM(model="llama3.1")
        
        # 2. Added the 'f' right before the quotes to inject the variables!
        prompt = f"""
        You are an expert AI assistant with deep knowledge in document analysis and information retrieval. Your role is to provide comprehensive, detailed, and well-structured answers based on the provided context and conversation history.

        ## Instructions:
        1. **Answer Length**: Provide detailed, comprehensive responses (aim for 3-5 paragraphs when appropriate)
        2. **Structure**: Use clear formatting with bullet points, numbered lists, or sections when helpful
        3. **Context Integration**: Synthesize information from the document context with conversation history
        4. **Follow-up Suggestions**: When relevant, suggest related questions or topics for deeper exploration
        5. **Accuracy**: Base your answer ONLY on the provided context and previous conversation

        ## Previous Conversation Context:
        {conversation_history}

        ## Document Context:
        {context}

        ## Current Question:
        {query}

        ## Response Guidelines:
        - If the answer is not found in the context, clearly state: "I am sorry, but I cannot find the answer to that in the provided documents. However, based on the available information, I can tell you that..."
        - Provide comprehensive explanations with examples when possible
        - Use clear, professional language while remaining accessible
        - Include relevant details, implications, and connections to other concepts
        - When appropriate, suggest follow-up questions that could provide additional value

        Please provide a thorough, well-structured response:
        """    
        response = llm.invoke(prompt)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)