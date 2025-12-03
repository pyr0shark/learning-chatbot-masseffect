"""
FastAPI Application
Main application serving API endpoints and static frontend.
"""

import os
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from litellm_client import AsyncLiteLLMClient
from rag_search import RAGSearch
from fact_extractor import FactExtractor
from indexer import Indexer

# Initialize FastAPI app
app = FastAPI(title="Mass Effect Lore Assistant")

# Mount static files from frontend directory
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Global components
client: Optional[AsyncLiteLLMClient] = None
rag_search: Optional[RAGSearch] = None
fact_extractor: Optional[FactExtractor] = None

# Session storage (in-memory)
conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
used_search_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
pending_facts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
seen_fact_hashes: Dict[str, set] = defaultdict(set)

# Background task queue (will be initialized in startup)
fact_discovery_queue: Optional[asyncio.Queue] = None


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str


class PendingFactsResponse(BaseModel):
    facts: List[Dict[str, Any]]


class ApproveFactRequest(BaseModel):
    session_id: str
    fact_id: str
    edited_fact: Optional[str] = None


class RejectFactRequest(BaseModel):
    session_id: str
    fact_id: str


# Startup event
@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    global client, rag_search, fact_extractor
    
    # Initialize LiteLLM client
    client = AsyncLiteLLMClient()
    
    # Initialize RAG search
    rag_search = RAGSearch(client)
    
    # Load or build index
    if os.path.exists("index.json"):
        print("Loading existing index...")
        rag_search.load_index("index.json")
    else:
        print("Index not found. Building from database.txt...")
        indexer = Indexer(client)
        await indexer.build_index("database.txt", "index.json")
        rag_search.load_index("index.json")
    
    print(f"Index loaded: {len(rag_search.chunks)} chunks")
    
    # Initialize fact extractor
    fact_extractor = FactExtractor(client, rag_search)
    
    # Initialize and start background fact worker
    global fact_discovery_queue
    fact_discovery_queue = asyncio.Queue()
    asyncio.create_task(fact_discovery_worker())
    
    print("Application started successfully")


# Background worker for fact discovery
async def fact_discovery_worker():
    """Background worker that processes fact-discovery jobs."""
    while True:
        try:
            session_id = await fact_discovery_queue.get()
            
            # Get session data
            history = conversation_history.get(session_id, [])
            used_results = used_search_results.get(session_id, [])
            seen_hashes = seen_fact_hashes.get(session_id, set())
            
            if not history or not used_results:
                continue
            
            # Process facts
            new_facts = await fact_extractor.process_facts(
                history,
                used_results,
                seen_hashes
            )
            
            # Add to pending facts
            for fact_text in new_facts:
                fact_id = str(uuid.uuid4())
                pending_facts[session_id].append({
                    "fact_id": fact_id,
                    "fact": fact_text,
                    "status": "pending",
                    "created_at": datetime.utcnow().isoformat()
                })
            
            fact_discovery_queue.task_done()
            
        except Exception as e:
            print(f"Error in fact discovery worker: {e}")
            await asyncio.sleep(1)


# Root endpoint - serve frontend
@app.get("/")
async def read_root():
    """Serve the frontend index.html."""
    return FileResponse("frontend/index.html")


# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG flow."""
    global rag_search, fact_extractor
    
    if not rag_search:
        raise HTTPException(status_code=500, detail="RAG search not initialized")
    
    # Validate input
    message = request.message.strip()
    session_id = request.session_id or "default"
    
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Step 1: Generate search terms
        history = conversation_history.get(session_id, [])
        search_terms = await rag_search.generate_search_terms(message, history)
        
        # Step 2: Vector search for each term
        search_results = await rag_search.search_multiple_terms(search_terms, k_per_term=10)
        
        # Step 3: Limit context chunks (max ~8000 tokens)
        context_chunks = rag_search.get_chunk_texts(search_results, max_tokens=8000)
        context_text = "\n\n".join(context_chunks)
        
        # Step 4: Update used search results (deduplicate by chunk_id)
        existing_chunk_ids = {chunk["chunk_id"] for chunk in used_search_results[session_id]}
        for chunk in search_results:
            if chunk["chunk_id"] not in existing_chunk_ids:
                used_search_results[session_id].append(chunk)
                existing_chunk_ids.add(chunk["chunk_id"])
        
        # Step 5: Generate answer
        history_text = ""
        if history:
            recent_history = history[-10:]  # Last 10 exchanges
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
        
        prompt = f"""You are a helpful Mass Effect lore assistant. Answer questions about the Mass Effect universe using the provided context and conversation history.

USER QUESTION: {message}

CONTEXT FROM KNOWLEDGE BASE:
{context_text}

{f'RECENT CONVERSATION HISTORY:\n{history_text}' if history_text else ''}

Instructions:
- Provide accurate, detailed answers based on the context
- If the context doesn't contain enough information, say so
- Reference specific characters, events, or locations when relevant
- Keep answers clear and well-structured
- If the user asks in another language, still respond in English (as the assistant)
"""
        
        answer = await client.text_completion(
            prompt=prompt,
            model_name="azure.gpt-5",
            max_tokens=15000,
            temperature=0.7
        )
        
        # Step 6: Update conversation history
        conversation_history[session_id].append({"role": "user", "content": message})
        conversation_history[session_id].append({"role": "assistant", "content": answer})
        
        # Step 7: Enqueue fact-discovery job (non-blocking)
        if fact_discovery_queue is not None:
            try:
                fact_discovery_queue.put_nowait(session_id)
            except asyncio.QueueFull:
                pass  # Skip if queue is full
        
        return ChatResponse(response=answer)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


# Get pending facts
@app.get("/api/facts/pending", response_model=PendingFactsResponse)
async def get_pending_facts(session_id: str):
    """Get pending facts for a session."""
    facts = [
        fact for fact in pending_facts.get(session_id, [])
        if fact["status"] == "pending"
    ]
    return PendingFactsResponse(facts=facts)


# Approve fact
@app.post("/api/facts/approve")
async def approve_fact(request: ApproveFactRequest):
    """Approve a fact and add it to the index."""
    global rag_search
    
    if not rag_search:
        raise HTTPException(status_code=500, detail="RAG search not initialized")
    
    # Find the fact
    session_facts = pending_facts.get(request.session_id, [])
    fact = next((f for f in session_facts if f["fact_id"] == request.fact_id), None)
    
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    
    if fact["status"] != "pending":
        raise HTTPException(status_code=400, detail="Fact already processed")
    
    # Use edited text if provided, otherwise original
    fact_text = request.edited_fact.strip() if request.edited_fact else fact["fact"]
    
    if not fact_text:
        raise HTTPException(status_code=400, detail="Fact text cannot be empty")
    
    try:
        # Generate embedding
        embeddings = await client.embedding(
            input_texts=[fact_text],
            model_name="azure.text-embedding-3-large",
            dimensions=1536
        )
        
        # Create chunk
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(fact_text))
        
        # Get next chunk_id
        existing_chunk_ids = [chunk["chunk_id"] for chunk in rag_search.chunks]
        next_chunk_id = max(existing_chunk_ids) + 1 if existing_chunk_ids else 0
        
        new_chunk = {
            "chunk_id": next_chunk_id,
            "text": fact_text,
            "token_count": token_count,
            "start_token": 0,
            "end_token": token_count,
            "embedding": embeddings[0]
        }
        
        # Add to index
        rag_search.add_chunk(new_chunk)
        
        # Update pending fact status
        fact["status"] = "approved"
        fact["approved_at"] = datetime.utcnow().isoformat()
        
        return {"status": "success", "message": "Fact approved and added to index"}
        
    except Exception as e:
        print(f"Error approving fact: {e}")
        raise HTTPException(status_code=500, detail=f"Error approving fact: {str(e)}")


# Reject fact
@app.post("/api/facts/reject")
async def reject_fact(request: RejectFactRequest):
    """Reject a pending fact."""
    session_facts = pending_facts.get(request.session_id, [])
    fact = next((f for f in session_facts if f["fact_id"] == request.fact_id), None)
    
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    
    if fact["status"] != "pending":
        raise HTTPException(status_code=400, detail="Fact already processed")
    
    # Mark as rejected
    fact["status"] = "rejected"
    fact["rejected_at"] = datetime.utcnow().isoformat()
    
    return {"status": "success", "message": "Fact rejected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

