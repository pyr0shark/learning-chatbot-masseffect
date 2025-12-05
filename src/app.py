"""
FastAPI Application
Main application serving API endpoints and static frontend.
"""

import os
import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    # Relative imports when used as a module
    from .litellm_client import AsyncLiteLLMClient
    from .rag_search import RAGSearch
    from .fact_extractor import FactExtractor
    from .indexer import Indexer
except ImportError:
    # Absolute imports when running as script
    from src.litellm_client import AsyncLiteLLMClient
    from src.rag_search import RAGSearch
    from src.fact_extractor import FactExtractor
    from src.indexer import Indexer

# Initialize FastAPI app
app = FastAPI(title="Mass Effect Lore Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

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
previously_extracted_facts: Dict[str, List[str]] = defaultdict(list)  # Track all candidate facts extracted in this session

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


class AllFactsResponse(BaseModel):
    approved: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]


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


# Immediate fact processing function
async def process_facts_for_session(session_id: str):
    """Process facts for a session immediately."""
    global fact_extractor
    
    logger.info(f"[FACT-DISCOVERY] Starting fact discovery for session: {session_id}")
    
    if not fact_extractor:
        logger.warning(f"[FACT-DISCOVERY] Fact extractor not available for session {session_id}")
        return
    
    try:
        # Get session data
        history = conversation_history.get(session_id, [])
        used_results = used_search_results.get(session_id, [])
        seen_hashes = seen_fact_hashes.get(session_id, set())
        
        logger.info(f"[FACT-DISCOVERY] Session data - History: {len(history)} messages, Used results: {len(used_results)} chunks, Seen hashes: {len(seen_hashes)}")
        
        if not history or not used_results:
            logger.info(f"[FACT-DISCOVERY] Skipping - insufficient data (history: {len(history)}, used_results: {len(used_results)})")
            return
        
        # Get previously extracted facts for this session
        prev_extracted = previously_extracted_facts.get(session_id, [])
        
        # Also include all facts from pending_facts (approved, rejected, or pending) to avoid re-extraction
        pending_facts_list = pending_facts.get(session_id, [])
        pending_fact_texts = [fact["fact"] for fact in pending_facts_list]
        prev_extracted = list(set(prev_extracted + pending_fact_texts))  # Combine and deduplicate
        
        logger.info(f"[FACT-DISCOVERY] Previously extracted facts in this session: {len(prev_extracted)} (including {len(pending_facts_list)} from pending_facts)")
        
        # Process facts
        logger.info(f"[FACT-DISCOVERY] Processing facts...")
        new_facts = await fact_extractor.process_facts(
            history,
            used_results,
            seen_hashes,
            prev_extracted
        )
        
        logger.info(f"[FACT-DISCOVERY] Found {len(new_facts)} new facts")
        for i, fact_text in enumerate(new_facts, 1):
            logger.info(f"[FACT-DISCOVERY] Fact {i}/{len(new_facts)}: {fact_text[:100]}...")
        
        # Add to pending facts and track as previously extracted
        for fact_text in new_facts:
            fact_id = str(uuid.uuid4())
            pending_facts[session_id].append({
                "fact_id": fact_id,
                "fact": fact_text,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            })
            # Track this fact as previously extracted to avoid extracting it again
            previously_extracted_facts[session_id].append(fact_text)
            logger.info(f"[FACT-DISCOVERY] Added fact to pending (ID: {fact_id}): {fact_text[:80]}...")
        
        logger.info(f"[FACT-DISCOVERY] Completed fact discovery for session {session_id} - {len(new_facts)} facts added to pending (total extracted in session: {len(previously_extracted_facts[session_id])})")
        
    except Exception as e:
        logger.error(f"[FACT-DISCOVERY] Error processing facts for session {session_id}: {e}", exc_info=True)

# Background worker for fact discovery (kept for compatibility, but may not be used)
async def fact_discovery_worker():
    """Background worker that processes fact-discovery jobs."""
    while True:
        try:
            session_id = await fact_discovery_queue.get()
            
            # Process facts using the same function
            await process_facts_for_session(session_id)
            
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
        logger.info(f"[CHAT] Session: {session_id} | User message: {message[:100]}...")
        
        # Step 1: Generate search terms
        history = conversation_history.get(session_id, [])
        logger.info(f"[CHAT] Step 1: Generating search terms (history length: {len(history)})")
        search_terms = await rag_search.generate_search_terms(message, history)
        logger.info(f"[CHAT] Step 1: Generated {len(search_terms)} search terms: {search_terms}")
        
        # Step 2: Vector search for each term
        logger.info(f"[CHAT] Step 2: Performing vector search for {len(search_terms)} terms")
        search_results = await rag_search.search_multiple_terms(search_terms, k_per_term=10)
        logger.info(f"[CHAT] Step 2: Found {len(search_results)} total search results")
        
        # Log unique chunk IDs found
        unique_chunk_ids = {chunk["chunk_id"] for chunk in search_results}
        logger.info(f"[CHAT] Step 2: Unique chunks found: {len(unique_chunk_ids)} chunks (IDs: {sorted(list(unique_chunk_ids))[:10]}...)")
        
        # Step 3: Get all context chunks (no token limit)
        logger.info(f"[CHAT] Step 3: Getting all context chunks (no token limit)")
        context_chunks = rag_search.get_chunk_texts(search_results, max_tokens=999999999)
        context_text = "\n\n".join(context_chunks)
        logger.info(f"[CHAT] Step 3: Selected {len(context_chunks)} context chunks for answer generation")
        
        # Step 4: Update used search results (deduplicate by chunk_id)
        existing_chunk_ids = {chunk["chunk_id"] for chunk in used_search_results[session_id]}
        new_chunks_count = 0
        for chunk in search_results:
            if chunk["chunk_id"] not in existing_chunk_ids:
                used_search_results[session_id].append(chunk)
                existing_chunk_ids.add(chunk["chunk_id"])
                new_chunks_count += 1
        logger.info(f"[CHAT] Step 4: Added {new_chunks_count} new chunks to used_search_results (total: {len(used_search_results[session_id])})")
        
        # Step 5: Generate answer
        history_text = ""
        if history:
            recent_history = history[-10:]  # Last 10 exchanges
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
        
        history_section = f"RECENT CONVERSATION HISTORY:\n{history_text}" if history_text else ""
        
        prompt = f"""You are a helpful Mass Effect lore assistant. Answer questions about the Mass Effect universe using the provided context and conversation history.

USER QUESTION: {message}

CONTEXT FROM KNOWLEDGE BASE:
{context_text}

{history_section}

Instructions:
- You are ONLY allowed to use knowledge that is explicitly stated in the CONTEXT FROM KNOWLEDGE BASE above
- You must NOT use any external knowledge, general knowledge, or information not present in the provided context
- Provide a complete, detailed, and comprehensive answer based ONLY on the context provided
- Answer the question fully using all relevant information from the context
- Do NOT start your response with phrases like "Short answer:" or "Brief answer:" - just provide the answer directly
- Do NOT add notes, disclaimers, or statements about what information is or is not in the knowledge base
- Simply answer the question using the information available in the context
- If the context doesn't contain enough information to fully answer the question, provide what you can from the context and do not mention what is missing
- Reference specific characters, events, or locations when relevant (only if they appear in the context)
- Keep answers clear, well-structured, and complete
- If the user asks in another language, still respond in English (as the assistant)
"""
        
        logger.info(f"[CHAT] Step 5: Generating answer with model azure.gpt-5-mini")
        answer = await client.text_completion(
            prompt=prompt,
            model_name="azure.gpt-5-mini",
            max_tokens=15000,
            temperature=0.7
        )
        logger.info(f"[CHAT] Step 5: Generated answer (length: {len(answer)} chars)")
        
        # Step 6: Update conversation history
        conversation_history[session_id].append({"role": "user", "content": message})
        conversation_history[session_id].append({"role": "assistant", "content": answer})
        logger.info(f"[CHAT] Step 6: Updated conversation history (total messages: {len(conversation_history[session_id])})")
        
        # Step 7: Process facts immediately in background (non-blocking)
        if fact_extractor is not None:
            logger.info(f"[CHAT] Step 7: Starting fact discovery for session {session_id}")
            # Start fact discovery immediately as a background task
            asyncio.create_task(process_facts_for_session(session_id))
        else:
            logger.warning(f"[CHAT] Step 7: Fact extractor not available, skipping fact discovery")
        
        logger.info(f"[CHAT] Session: {session_id} | Response sent successfully")
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


# Get all facts (approved and rejected)
@app.get("/api/facts/all", response_model=AllFactsResponse)
async def get_all_facts():
    """Get all approved and rejected facts from all sessions."""
    approved_facts = []
    rejected_facts = []
    
    # Collect all facts from all sessions
    for session_id, facts in pending_facts.items():
        for fact in facts:
            if fact.get("status") == "approved":
                approved_facts.append({
                    **fact,
                    "session_id": session_id
                })
            elif fact.get("status") == "rejected":
                rejected_facts.append({
                    **fact,
                    "session_id": session_id
                })
    
    # Sort by date (most recent first)
    approved_facts.sort(key=lambda x: x.get("approved_at", ""), reverse=True)
    rejected_facts.sort(key=lambda x: x.get("rejected_at", ""), reverse=True)
    
    return AllFactsResponse(approved=approved_facts, rejected=rejected_facts)


if __name__ == "__main__":
    import uvicorn
    import sys
    from pathlib import Path
    # Add parent directory to path for running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    # Import with absolute path for script execution
    from src.litellm_client import AsyncLiteLLMClient
    from src.rag_search import RAGSearch
    from src.fact_extractor import FactExtractor
    from src.indexer import Indexer
    uvicorn.run(app, host="0.0.0.0", port=8000)

