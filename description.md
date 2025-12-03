# Mass Effect Lore RAG Chatbot – Agent Specification

You are helping me build and maintain a **Mass Effect lore chatbot** with:

- A **Python FastAPI backend** (one server, one web app)
- A **JavaScript frontend** served as static files by that same backend
- A **RAG (Retrieval-Augmented Generation) pipeline** over `database.txt` + incremental user-approved lore
- **Azure OpenAI via LiteLLM**, using **async APIs only**
- Packaged as a **single Docker container** that can run locally and be deployed as an **Azure Container App**

There must be **no second server or separate backend process**. Everything lives behind a single FastAPI app inside one container.

---

## 0. Containerization & Deployment

- The project must build into **one Docker image** that:
  - Runs locally via `docker run -p 8000:8000 ...`
  - Listens on `0.0.0.0:8000` inside the container
- Primary cloud deployment target: **Azure Container Apps**.
  - Container Apps will route HTTP traffic to port `8000`.
  - Configuration (e.g. `OPENAI_API_BASE`, `OPENAI_API_KEY`) must be via **environment variables**.
- All code and static files (including `frontend/`) must be included in the image.
- The app must not rely on any host-mounted paths by default; optional volume mounting for `index.json` persistence is allowed but not required by this spec.

---

## 1. Purpose & Behavior

The chatbot is a **Mass Effect lore assistant**. It should:

- Answer questions about:
  - Characters, events, locations, technology, factions, timeline, etc.
- Ground answers in a **vector index** built from `database.txt` (`index.json`).
- Keep **conversation history per session** so follow-ups are contextual.
- Detect when users mention **new lore facts** that are not in the index:
  - Ask the user to **approve/edit** those facts.
  - Only then add them to the index.

All prompts to LLMs must be in **English**, even if the user talks in another language.

---

## 2. Tech Stack & Project Layout

### 2.1. Backend

- **Python**
- **FastAPI** as the web framework
- **Gunicorn** + **Uvicorn worker** for production (inside Docker / Azure Container Apps)
- **LiteLLM** for calling Azure OpenAI
- **tiktoken** for tokenization (chunking & token counts)
- **hnswlib** + **numpy** for vector search (HNSW index)
- Storage:
  - `database.txt` → source corpus
  - `index.json` → serialized embedding index (including dynamically added facts)

### 2.2. Frontend

Served from a `frontend/` folder by the FastAPI app:

- `frontend/index.html`
  - Single-page UI titled **“Mass Effect Lore Assistant”**
  - Initial message introduces the assistant as a Mass Effect lore expert
- `frontend/styles.css`
  - Chat layout + styling
  - UI for fact approval cards (fact text, edit, approve, reject)
- `frontend/chatbot.js`
  - Sends chat requests to backend API
  - Manages `session_id`
  - Polls for pending facts and renders them as they appear

### 2.3. Root Structure (conceptual)

At the repo root:

- `app.py`  
  FastAPI app:
  - Chat API (`/api/chat`)
  - Fact-approval APIs
  - Serves static frontend files
  - Initializes RAG & HNSW on startup
- `litellm_client/`
  - `__init__.py`
  - `client.py` – **async-only** LiteLLM wrapper
- `indexer.py` – builds `index.json` from `database.txt`
- `rag_search.py` – RAG + HNSW search logic
- `fact_extractor.py` – conversation-based fact extraction logic
- `frontend/` – JS UI (as above)
- `database.txt` – base Mass Effect lore text
- `index.json` – vector index file
- `requirements.txt` – Python deps
- `startup.sh` or equivalent CMD for Docker (e.g. gunicorn command)

Do **not** introduce a second backend server or process.

---

## 3. LiteLLM Client (Async-Only)

Create a small library, e.g. `litellm_client.AsyncLiteLLMClient`, that wraps LiteLLM calls.

### 3.1. Configuration

Use **environment variables** only:

- `OPENAI_API_BASE` (e.g. `https://genai-sharedservice-emea.pwc.com`)
- `OPENAI_API_KEY`

Never hard-code keys in code or config.

### 3.2. Models & Settings

- **Search term generation**  
  - Model: `"azure.gpt-5-nano"`
  - `max_tokens`: **20_000**
  - `temperature`: ~`0.3` (precise)

- **Answer generation**  
  - Model: `"azure.gpt-5"`
  - `max_tokens`: **15_000**
  - `temperature`: ~`0.7` (more expressive but still grounded)

- **Embeddings**  
  - Model: `"azure.text-embedding-3-large"`
  - `dimensions`: `1536`

### 3.3. APIs

Use **async LiteLLM APIs only**:

- Async text completion (e.g. `atext_completion`) for:
  - search term generation
  - final answers
  - fact extraction
  - “fact already in index?” checks
- Async embedding (e.g. `aembedding`) for:
  - chunk embeddings
  - query embeddings
  - fact embeddings

No synchronous LiteLLM calls.

---

## 4. Indexing & HNSW

### 4.1. Base Index (`indexer.py`)

Index is generated from `database.txt` via **rolling window chunking**:

- Tokenization: via `tiktoken`
- Chunk size: **256 tokens**
- Overlap: **128 tokens** (sliding / rolling window)

For each chunk:

- Compute embedding with `"azure.text-embedding-3-large"` (1536-dim)
- Store in `index.json` as:

```json
{
  "metadata": {
    "source_file": "database.txt",
    "chunk_size": 256,
    "overlap": 128,
    "total_chunks": 0,
    "total_tokens": 0
  },
  "chunks": [
    {
      "chunk_id": 0,
      "text": "...",
      "token_count": 256,
      "start_token": 0,
      "end_token": 256,
      "embedding": [0.0]
    }
  ]
}
(Actual values will differ; above is structural.)

index.json is the single source of truth for the index, including later user-approved facts.

4.2. HNSW Index at App Startup
On FastAPI startup:

Load index.json.

Build an HNSW index (cosine similarity) with hnswlib from all chunk embeddings.

Keep HNSW index in memory for fast search.

When new facts are approved, they are:

Appended to index.json.

Added to the in-memory structure.

HNSW is rebuilt or updated accordingly.

5. Sessions, Conversation History & RAG Flow
5.1. Sessions
Each browser tab must use a new session on refresh:

On page load, frontend generates a fresh session_id, e.g. session_<timestamp>_<random>.

Do not reuse from localStorage; each refresh begins a new conversation.

session_id is sent with every API call:

/api/chat

Fact APIs

Pending-facts fetches

Per-session data (kept in memory on backend):

conversation_history[session_id]

used_search_results[session_id]

pending_facts[session_id]

Optionally seen_fact_hashes[session_id]

Global persistent data:

index.json only.

5.2. Conversation History
For each session, maintain a list:

python
Code kopiëren
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
]
Keep only a recent window (e.g. last ~10 exchanges) if needed for token limits.

5.3. Main RAG Answering Flow (Fast Path)
When POST /api/chat is called:

Read session & message

Use session_id from request (or "default" as fallback).

Strip and validate the user message.

Generate search terms (GPT-5-nano)

Use "azure.gpt-5-nano" with an English prompt including:

User question.

Recent conversation history.

Output: 3–5 short, specific search terms.

Async vector search

For each search term:

Embed with "azure.text-embedding-3-large".

Query HNSW for top-k chunks.

Merge and deduplicate retrieved chunks.

Append these chunks to used_search_results[session_id].

Answer generation (GPT-5)

Use "azure.gpt-5" with an English prompt:

“You are a helpful Mass Effect lore assistant…”

Include:

User question.

Selected context chunks (limited to a safe max).

Recent conversation history.

Produce a coherent, lore-accurate answer.

Save history & schedule fact job

Append {user, assistant} messages to conversation_history[session_id].

Enqueue a fact-discovery job for this session_id (see section 6).

Return the answer to the frontend immediately.

The RAG flow should be as fast as reasonably possible and must not wait for fact-discovery.

6. Fact-Discovery Pipeline (Background Sidecar)
The fact-discovery pipeline is separate from the chat flow. It may finish later, and its results appear in the UI asynchronously as “fact cards”.

6.1. Triggering
On every new user message (inside /api/chat):

After computing the RAG answer and updating history & used search results:

Enqueue a fact-discovery job for that session_id into an async queue.

A background worker (task started at app startup) consumes jobs and runs the fact pipeline.

6.2. Inputs to Fact Pipeline
For a given session_id, the worker uses:

conversation_history[session_id]
Entire conversation so far in this session.

used_search_results[session_id]
The union of all chunks that have been used in any RAG answer so far for this session.

These two sets are compared to identify new facts the user has introduced that are not in the used search results.

6.3. Step 1: Candidate Fact Extraction (Conversation vs Used Results)
Call "azure.gpt-5-nano" with an English prompt instructing it to:

Read:

Full conversation history.

Texts of all used search result chunks.

Extract factual statements that:

Are about Mass Effect lore (characters, events, locations, technology, history, factions, etc.).

Are explicitly stated in the conversation (not invented).

Are not present in the used search results (even if paraphrased).

Are not questions, opinions, or speculation.

Are clear, specific, and self-contained.

Are genuinely new information relative to the used search results.

The result is a list of candidate facts that:

The user has mentioned, and

Were not covered by the search results used so far in this session.

Deduplicate candidate facts against:

Existing pending_facts[session_id].

Already approved/rejected facts (optional).

Normalized hash of the fact text.

6.4. Step 2: Index Presence Check (Top-10 + GPT-5-nano)
Each candidate fact must be checked against the global index (index.json) to see if it is truly new.

For each fact:

Nearest neighbor search

Embed fact with "azure.text-embedding-3-large".

Query HNSW for the top 10 nearest chunks.

Collect those chunk texts as the “candidate index context” for this fact.

GPT-5-nano boolean check

Call "azure.gpt-5-nano" with an English prompt that:

Shows:

The fact as a claim.

The 10 closest index chunks.

Asks:

“Does the information in these chunks already contain this fact, even if paraphrased?”

Requires a strict boolean output (true / false, or an equivalent that can be parsed).

Interpretation

If GPT-5-nano says true →
Fact is considered already present in the index → discard from new-fact pipeline.

If GPT-5-nano says false →
Fact is treated as genuinely new to the index → proceed.

This LLM-on-top-10 pattern is the authoritative check for “already in index?”.

6.5. Step 3: Pending Facts Buffer (Per Session)
All facts that pass the index check are stored in:

pending_facts[session_id] as entries with:

id (fact_id)

text (original fact text)

created_at

status = "pending"

No change is made to index.json here.

The frontend polls an endpoint like:

GET /api/facts/pending?session_id=...

to retrieve these pending facts and render them as fact cards in the UI. These may appear at any time, not tied to sending/receiving a particular message.

6.6. Step 4: User Approval / Rejection & Index Update
The UI for each pending fact:

Shows the fact text.

Allows editing the text.

Has:

Approve button.

Reject button.

Approve flow:

Frontend sends POST /api/facts/approve with:

session_id

fact_id

edited_text (optional)

Backend:

Takes approved text.

Embeds with "azure.text-embedding-3-large".

Appends it as a new chunk into the index structure.

Writes updated index back to index.json.

Rebuilds / updates HNSW to include the new chunk.

Marks the fact as approved in pending_facts[session_id].

Reject flow:

Frontend sends POST /api/facts/reject with:

session_id

fact_id

Backend:

Marks the fact as rejected and/or removes it from pending_facts[session_id].

Does not touch index.json or HNSW.

6.7. Critical Constraints
The fact-discovery pipeline is fully decoupled from the main RAG answering path:

It runs via background worker(s).

It must not block /api/chat.

Facts are only added to index.json when:

A user explicitly approves a pending fact via the API.

No fact is ever added:

Automatically at startup.

During extraction.

During index-presence checks.

Without explicit user approval.

7. Frontend Behavior Summary
On load:

Generate a new session_id (per refresh).

Show initial lore assistant message.

When user sends a message:

POST to /api/chat with { message, session_id }.

Display bot’s response from response field.

Start or restart a short polling window that calls:

GET /api/facts/pending?session_id=... every few seconds.

Render any new pending facts as cards.

Fact cards:

Editable text area.

Approve/Reject buttons.

Approve → call /api/facts/approve.

Reject → call /api/facts/reject.

Facts can appear anytime after a message, not necessarily aligned with responses.

8. Coding Guidelines
When modifying or generating code for this project, you must:

Preserve the described architecture:

Single FastAPI app.

JS frontend served by backend.

Async LiteLLM client.

RAG via HNSW over index.json.

Background fact-discovery pipeline with user-approved index updates.

Packaged as a single Docker container suitable for Azure Container Apps.

Use async for all model calls.

Keep code reasonably clean:

Use type hints.

Use small, focused functions.

Add docstrings for core components (RAGSearch, FactExtractor, fact worker, etc.).

Maintain the constraints:

All LLM prompts in English.

No automatic fact insertion into the index without user approval.

No second server or separate backend process.

This document is the canonical specification for how the Cursor agent should shape and maintain the codebase and containerized deployment.
