"""
FastAPI Application
Main application serving API endpoints and static frontend.
"""

import os
import asyncio
import uuid
import logging
import re
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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json

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

# SSE event queues for pushing facts to clients
fact_event_queues: Dict[str, asyncio.Queue] = {}

# Background task queue (will be initialized in startup)
fact_discovery_queue: Optional[asyncio.Queue] = None


# Helper function to convert numbers to superscript
def number_to_superscript(num: int) -> str:
    """Convert a number to superscript format."""
    superscript_digits = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    return ''.join(superscript_digits.get(digit, digit) for digit in str(num))


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    references: List[Dict[str, Any]]


class PendingFactsResponse(BaseModel):
    facts: List[Dict[str, Any]]


class ApproveFactRequest(BaseModel):
    session_id: str
    fact_id: str
    edited_fact: Optional[str] = None


class RejectFactRequest(BaseModel):
    session_id: str
    fact_id: str


class DeleteFactRequest(BaseModel):
    fact_text: str


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
        
        # Initialize seen_hashes for this session if it doesn't exist
        if session_id not in seen_fact_hashes:
            seen_fact_hashes[session_id] = set()
        seen_hashes = seen_fact_hashes[session_id]
        
        # Add hashes of all facts identified in this session to seen_hashes
        # This includes: pending facts, previously extracted facts, and facts that were already in the index
        pending_facts_list = pending_facts.get(session_id, [])
        prev_extracted = previously_extracted_facts.get(session_id, [])
        
        # Add all facts from this session to seen_hashes
        all_session_facts = [fact["fact"] for fact in pending_facts_list] + prev_extracted
        for fact_text in all_session_facts:
            fact_hash = fact_extractor._hash_fact(fact_text)
            seen_hashes.add(fact_hash)
        
        logger.info(f"[FACT-DISCOVERY] Session data - History: {len(history)} messages, Used results: {len(used_results)} chunks, Seen hashes: {len(seen_hashes)} (including {len(all_session_facts)} facts from this session)")
        
        if not history or not used_results:
            logger.info(f"[FACT-DISCOVERY] Skipping - insufficient data (history: {len(history)}, used_results: {len(used_results)})")
            return
        
        # Get previously extracted facts for this session (for the extract_candidate_facts prompt)
        prev_extracted = previously_extracted_facts.get(session_id, [])
        
        # Also include all facts from pending_facts (approved, rejected, or pending) to avoid re-extraction
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
            fact_data = {
                "fact_id": fact_id,
                "fact": fact_text,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }
            pending_facts[session_id].append(fact_data)
            # Track this fact as previously extracted to avoid extracting it again
            previously_extracted_facts[session_id].append(fact_text)
            logger.info(f"[FACT-DISCOVERY] Added fact to pending (ID: {fact_id}): {fact_text[:80]}...")
            
            # Push fact to SSE stream for this session
            if session_id in fact_event_queues:
                try:
                    await fact_event_queues[session_id].put(fact_data)
                    logger.info(f"[FACT-DISCOVERY] Pushed fact {fact_id} to SSE stream for session {session_id}")
                except Exception as e:
                    logger.warning(f"[FACT-DISCOVERY] Failed to push fact to SSE stream: {e}")
        
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
        
        # Step 3: Get all context chunks (no token limit) and prepare for references
        logger.info(f"[CHAT] Step 3: Getting all context chunks (no token limit)")
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Get chunks with their full metadata for references
        context_chunks_with_refs = []
        total_tokens = 0
        for chunk in search_results:
            chunk_tokens = len(encoding.encode(chunk["text"]))
            if total_tokens + chunk_tokens > 999999999:  # Same limit as before
                break
            context_chunks_with_refs.append(chunk)
            total_tokens += chunk_tokens
        
        # Number the chunks and format context with reference numbers
        numbered_context_parts = []
        reference_list = []
        ref_num_to_chunk = {}  # Map reference number to full chunk text
        
        for ref_num, chunk in enumerate(context_chunks_with_refs, start=1):
            # Format chunk with reference number
            numbered_context_parts.append(f"[{ref_num}] {chunk['text']}")
            
            # Store mapping from reference number to chunk text
            ref_num_to_chunk[ref_num] = chunk['text']
            
            # Create reference entry (use first 100 chars of text as description)
            ref_text = chunk['text'].strip()
            ref_preview = ref_text[:100] + "..." if len(ref_text) > 100 else ref_text
            reference_list.append({
                "num": ref_num,
                "text": ref_preview,
                "superscript": number_to_superscript(ref_num)
            })
        
        # Add a clear header explaining the sequential numbering
        context_header = f"IMPORTANT: The following {len(context_chunks_with_refs)} context chunks are numbered SEQUENTIALLY from 1 to {len(context_chunks_with_refs)}. Use ONLY these sequential numbers (1, 2, 3, ..., {len(context_chunks_with_refs)}) as reference numbers. Do NOT use any other numbers.\n\n"
        context_text = context_header + "\n\n".join(numbered_context_parts)
        logger.info(f"[CHAT] Step 3: Selected {len(context_chunks_with_refs)} context chunks for answer generation (numbered 1-{len(context_chunks_with_refs)})")
        ref_mapping_str = ', '.join([f'Ref {ref["num"]}->{ref["superscript"]}' for ref in reference_list[:10]])
        if len(reference_list) > 10:
            ref_mapping_str += '...'
        logger.info(f"[CHAT] Step 3: Reference mapping: {ref_mapping_str}")
        
        # Step 4: Update used search results (deduplicate by chunk_id)
        existing_chunk_ids = {chunk["chunk_id"] for chunk in used_search_results[session_id]}
        new_chunks_count = 0
        for chunk in search_results:
            if chunk["chunk_id"] not in existing_chunk_ids:
                used_search_results[session_id].append(chunk)
                existing_chunk_ids.add(chunk["chunk_id"])
                new_chunks_count += 1
        logger.info(f"[CHAT] Step 4: Added {new_chunks_count} new chunks to used_search_results (total: {len(used_search_results[session_id])})")
        
        # Step 4.5: Start fact discovery immediately in background (non-blocking)
        # This starts as soon as we have the search results, before generating the answer
        if fact_extractor is not None:
            logger.info(f"[CHAT] Step 4.5: Starting fact discovery immediately for session {session_id}")
            # Update conversation history with user message first (needed for fact extraction)
            conversation_history[session_id].append({"role": "user", "content": message})
            # Start fact discovery immediately as a background task
            asyncio.create_task(process_facts_for_session(session_id))
        else:
            logger.warning(f"[CHAT] Step 4.5: Fact extractor not available, skipping fact discovery")
        
        # Step 5: Generate answer
        history_text = ""
        if history:
            recent_history = history[-10:]  # Last 10 exchanges
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
        
        history_section = f"RECENT CONVERSATION HISTORY:\n{history_text}" if history_text else ""
        
        # Create instructions for superscript references
        if len(reference_list) <= 20:
            # Show all references if 20 or fewer
            ref_instructions = "\n".join([
                f"- Reference {ref['num']} → use superscript {ref['superscript']}"
                for ref in reference_list
            ])
        else:
            # Show first 10 and last 10 if more than 20
            ref_instructions = "\n".join([
                f"- Reference {ref['num']} → use superscript {ref['superscript']}"
                for ref in reference_list[:10]
            ])
            ref_instructions += f"\n- ... (references 11 through {len(reference_list) - 10} follow the same pattern) ..."
            ref_instructions += "\n" + "\n".join([
                f"- Reference {ref['num']} → use superscript {ref['superscript']}"
                for ref in reference_list[-10:]
            ])
        
        ref_instructions = "Reference Number Mapping:\n" + ref_instructions + "\n\nRemember: Use ONLY these sequential reference numbers. Do NOT use any other numbers."
        
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

IMPORTANT - CITATION REQUIREMENTS:
- The context chunks are numbered sequentially as [1], [2], [3], etc. in the order they appear
- You MUST use ONLY these sequential reference numbers (1, 2, 3, 4, ...) - do NOT use any other numbers
- You MUST include superscript reference numbers throughout your response wherever you use information from the context
- Use superscript numbers (¹, ², ³, etc.) immediately after sentences or phrases that use information from the corresponding reference
- The superscript number MUST match the sequential reference number (Reference 1 = ¹, Reference 2 = ², Reference 3 = ³, etc.)
- You can use multiple references in the same sentence if needed (e.g., "Shepard was a Spectre¹ and fought the Reapers²")
- At the end of your response, you MUST include a list of reference numbers you used, formatted as a Python-style list: [1, 4, 8] (for example, if you used references 1, 4, and 8)
- Do NOT include a "References:" section - only include the list of numbers at the end

{ref_instructions}
"""
        
        logger.info(f"[CHAT] Step 5: Generating answer with model azure.gpt-5-mini")
        answer = await client.text_completion(
            prompt=prompt,
            model_name="azure.gpt-5-mini",
            max_tokens=15000,
            temperature=0.7
        )
        logger.info(f"[CHAT] Step 5: Generated answer (length: {len(answer)} chars)")
        
        # Step 5.5: Extract reference list from answer and build JSON array
        answer_lower = answer.lower()
        
        # Remove any "References:" section if it exists
        if "references:" in answer_lower:
            # Find where "References:" section starts and remove everything from there
            for i in range(len(answer)):
                if answer[i:].lower().startswith("references:"):
                    answer = answer[:i].rstrip()
                    break
        
        # Extract reference list in format [1, 4, 8] from the answer
        ref_list_pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        ref_list_match = re.search(ref_list_pattern, answer)
        
        if ref_list_match:
            # Extract the numbers from the list
            ref_list_str = ref_list_match.group(1)  # Get the content inside brackets
            # Parse the numbers (handle spaces around commas)
            ref_numbers = [int(num.strip()) for num in ref_list_str.split(',')]
            # Remove the list from the answer text
            answer = re.sub(ref_list_pattern, '', answer).rstrip()
        else:
            # Fallback: detect which references were actually used by looking for superscripts
            used_ref_nums = set()
            for ref in reference_list:
                # Check if the superscript appears in the answer
                if ref['superscript'] in answer:
                    used_ref_nums.add(ref['num'])
            
            if used_ref_nums:
                ref_numbers = sorted(list(used_ref_nums))
            else:
                # Last resort: use all references
                ref_numbers = sorted([ref['num'] for ref in reference_list])
        
        # Build the references JSON array using the extracted reference numbers
        references_json = []
        for ref_num in ref_numbers:
            if ref_num in ref_num_to_chunk:
                references_json.append({
                    "reference": ref_num,
                    "chunk": ref_num_to_chunk[ref_num]
                })
        
        # Step 6: Update conversation history
        conversation_history[session_id].append({"role": "user", "content": message})
        conversation_history[session_id].append({"role": "assistant", "content": answer})
        logger.info(f"[CHAT] Step 6: Updated conversation history (total messages: {len(conversation_history[session_id])})")
        
        logger.info(f"[CHAT] Session: {session_id} | Response sent successfully with {len(references_json)} references")
        return ChatResponse(response=answer, references=references_json)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


# Get pending facts (kept for backwards compatibility, but SSE is preferred)
@app.get("/api/facts/pending", response_model=PendingFactsResponse)
async def get_pending_facts(session_id: str):
    """Get pending facts for a session."""
    logger.info(f"[FACTS-API] Getting pending facts for session: {session_id}")
    session_facts = pending_facts.get(session_id, [])
    logger.info(f"[FACTS-API] Total facts in session: {len(session_facts)}")
    facts = [
        fact for fact in session_facts
        if fact.get("status") == "pending"
    ]
    logger.info(f"[FACTS-API] Returning {len(facts)} pending facts")
    return PendingFactsResponse(facts=facts)


# SSE endpoint for real-time fact updates
@app.get("/api/facts/stream")
async def stream_facts(session_id: str):
    """Server-Sent Events stream for real-time fact updates."""
    async def event_generator():
        # Create queue for this session if it doesn't exist
        if session_id not in fact_event_queues:
            fact_event_queues[session_id] = asyncio.Queue()
        
        queue = fact_event_queues[session_id]
        
        # Send any existing pending facts first
        session_facts = pending_facts.get(session_id, [])
        pending = [f for f in session_facts if f.get("status") == "pending"]
        for fact in pending:
            yield f"data: {json.dumps(fact)}\n\n"
        
        # Keep connection alive and send new facts as they arrive
        try:
            while True:
                # Wait for new fact with timeout to send keepalive
                try:
                    fact = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(fact)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            logger.info(f"[SSE] Connection closed for session {session_id}")
        finally:
            # Clean up queue when connection closes
            if session_id in fact_event_queues:
                del fact_event_queues[session_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
            "embedding": embeddings[0],
            "source": "user_approved",  # Mark chunks added via user approval
            "created_at": fact.get("created_at", datetime.utcnow().isoformat()),  # Use original creation time if available
            "approved_at": datetime.utcnow().isoformat(),
            "session_id": request.session_id
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


# Delete fact from index
@app.post("/api/facts/delete")
async def delete_fact(request: DeleteFactRequest):
    """Delete a fact from the index."""
    global rag_search
    
    if not rag_search:
        raise HTTPException(status_code=500, detail="RAG search not initialized")
    
    try:
        # Remove from index
        removed = rag_search.remove_chunk_by_text(request.fact_text)
        
        if not removed:
            raise HTTPException(status_code=404, detail="Fact not found in index")
        
        # Also update status in pending_facts if it exists
        for session_id, facts in pending_facts.items():
            for fact in facts:
                if fact.get("fact") == request.fact_text and fact.get("status") == "approved":
                    fact["status"] = "deleted"
                    fact["deleted_at"] = datetime.utcnow().isoformat()
                    break
        
        return {"status": "success", "message": "Fact deleted from index"}
        
    except ValueError as e:
        # ValueError is raised when trying to delete database.txt chunks
        logger.warning(f"Cannot delete fact: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting fact: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting fact: {str(e)}")


# Reset index to database.txt only
@app.post("/api/index/reset")
async def reset_index():
    """Reset the index to only contain database.txt chunks, removing all user-approved facts."""
    global rag_search
    
    if not rag_search:
        raise HTTPException(status_code=500, detail="RAG search not initialized")
    
    try:
        # Remove all user_approved chunks
        removed_count = rag_search.reset_to_database_only()
        
        # Also mark all approved facts in pending_facts as deleted
        deleted_facts = 0
        for session_id, facts in pending_facts.items():
            for fact in facts:
                if fact.get("status") == "approved":
                    fact["status"] = "deleted"
                    fact["deleted_at"] = datetime.utcnow().isoformat()
                    deleted_facts += 1
        
        logger.info(f"[INDEX-RESET] Removed {removed_count} chunks from index, marked {deleted_facts} facts as deleted")
        
        return {
            "status": "success",
            "message": f"Index reset successfully. Removed {removed_count} user-approved chunks.",
            "chunks_removed": removed_count,
            "facts_marked_deleted": deleted_facts,
            "remaining_chunks": len(rag_search.chunks)
        }
        
    except Exception as e:
        logger.error(f"Error resetting index: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting index: {str(e)}")


# Get all facts (approved and rejected)
@app.get("/api/facts/all", response_model=AllFactsResponse)
async def get_all_facts():
    """Get all approved and rejected facts from all sessions and from the index."""
    global rag_search
    
    approved_facts = []
    rejected_facts = []
    seen_approved_texts = set()  # To deduplicate
    
    # First, collect all approved facts from the index (user_approved chunks)
    if rag_search:
        for chunk in rag_search.chunks:
            if chunk.get("source") == "user_approved":
                fact_text = chunk.get("text", "").strip()
                if fact_text and fact_text not in seen_approved_texts:
                    approved_facts.append({
                        "fact_id": f"index_{chunk.get('chunk_id')}",
                        "fact": fact_text,
                        "status": "approved",
                        "approved_at": chunk.get("approved_at", chunk.get("created_at", "")),
                        "created_at": chunk.get("created_at", chunk.get("approved_at", "")),
                        "session_id": chunk.get("session_id", "unknown"),
                        "source": "index"
                    })
                    seen_approved_texts.add(fact_text)
    
    # Then, collect all facts from pending_facts (in-memory session facts)
    for session_id, facts in pending_facts.items():
        for fact in facts:
            fact_text = fact.get("fact", "").strip()
            
            if fact.get("status") == "approved":
                # Only add if not already in index (deduplicate)
                if fact_text not in seen_approved_texts:
                    approved_facts.append({
                        **fact,
                        "session_id": session_id,
                        "source": "session"
                    })
                    seen_approved_texts.add(fact_text)
            elif fact.get("status") == "rejected":
                rejected_facts.append({
                    **fact,
                    "session_id": session_id,
                    "source": "session"
                })
    
    # Sort by date (most recent first)
    approved_facts.sort(key=lambda x: x.get("approved_at", x.get("created_at", "")), reverse=True)
    rejected_facts.sort(key=lambda x: x.get("rejected_at", x.get("created_at", "")), reverse=True)
    
    logger.info(f"[FACTS-API] Returning {len(approved_facts)} approved facts ({len([f for f in approved_facts if f.get('source') == 'index'])} from index, {len([f for f in approved_facts if f.get('source') == 'session'])} from sessions) and {len(rejected_facts)} rejected facts")
    
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

