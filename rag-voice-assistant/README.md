# rag-voice-assistant

Prototype RAG pipeline for a voice-based political campaign assistant. Upload campaign documents, ask questions via text or mic, get answers back with audio.

Still early — see known issues below.

---

## demo

<!-- drop screenshot here -->
![ui](docs/screenshot.png)

<!-- swap this with actual link when recorded -->
[demo video](https://your-link-here)

---

## what it does

Campaign documents (manifestos, district info, candidate profiles, FAQs) get chunked and embedded into a local vector store. When a user asks something — via text or voice — the system retrieves the most relevant chunks and passes them to an LLM to generate a grounded answer. Response is spoken back via neural TTS.

```
[mic / text input]
      │
      ▼
 Whisper STT
      │
      ▼
 embed query → ChromaDB similarity search → top-k chunks
                                                  │
                                                  ▼
                                        build prompt + context
                                                  │
                                                  ▼
                                          gpt-4o-mini
                                                  │
                                                  ▼
                                    script detection (Devanagari?)
                                         │               │
                                         ▼               ▼
                                   Hindi TTS       English TTS
                                (MadhurNeural)  (NeerjaNeural)
```

ingestion (offline):
```
upload doc → parse (pdf/docx/txt/md) → chunk (500t, 50 overlap) → embed → ChromaDB
```

---

## stack

- **api** — FastAPI
- **vector db** — ChromaDB (local persistent)
- **embeddings** — sentence-transformers `all-MiniLM-L6-v2`
- **stt** — openai-whisper `small`
- **llm** — openai `gpt-4o-mini`
- **tts** — edge-tts (Microsoft neural voices)
- **chunking** — LangChain RecursiveCharacterTextSplitter

---

## setup

```bash
conda create -n rag-voice-assistant python=3.11
conda activate rag-voice-assistant
pip install -r requirements.txt
```

add `.env`:
```
OPENAI_API_KEY=sk-...
```

run:
```bash
python -m uvicorn main:app --reload
```

open `http://127.0.0.1:8000` in Chrome.

---

## endpoints

| method | path | description |
|---|---|---|
| GET | `/health` | sanity check |
| POST | `/upload` | ingest a document with metadata |
| POST | `/retrieve` | raw semantic search, returns chunks |
| POST | `/query` | full rag — retrieve + llm response |
| POST | `/voice-query` | audio in → whisper → rag → tts audio out |

quick test:
```bash
# upload
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_docs/krishna_district.txt" \
  -F "district=Krishna" -F "category=schemes" -F "topic=farmer welfare"

# query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what schemes exist for farmers in krishna district?"}'
```

---

## sample docs

five documents in `sample_docs/` covering krishna district schemes, candidate profile, general voter FAQ, agriculture manifesto, and vijayawada-specific FAQ. upload all before testing.

---

## project structure

```
├── main.py           # fastapi app + all routes
├── ingest.py         # parse → chunk → embed → store
├── retriever.py      # semantic search wrapper
├── llm.py            # prompt builder + openai call
├── config.py         # constants, env vars
├── static/
│   └── index.html    # chat ui
├── sample_docs/      # test documents
├── data/             # uploaded files land here
└── chroma_db/        # persisted vector store
```

---

## scaling this

current setup is single-process, local, no auth — fine for a demo. production path:

- **vector db** — swap ChromaDB for Pinecone or Qdrant. ChromaDB doesn't handle concurrent writes well and has no replication
- **embeddings** — move to a hosted embeddings endpoint (OpenAI `text-embedding-3-small` or a dedicated GPU instance) — local sentence-transformers blocks the event loop
- **whisper** — replace local whisper with a streaming STT service (Deepgram, AssemblyAI) that emits partial transcripts. retrieval can start before the user finishes speaking, cutting ~800ms off response latency
- **llm** — stream the OpenAI response token by token back to the client instead of waiting for the full completion
- **document ingestion** — move to a background job queue (Celery + Redis) so large PDF uploads don't block the API
- **storage** — documents currently land on local disk. S3 or GCS for anything beyond single-instance
- **auth** — zero auth right now. real deployment needs at minimum API key validation on every endpoint

none of this is needed for the prototype but the architecture doesn't need to change — just swap the components out.

---

## known issues / todo

- similarity threshold set low (0.1) — small doc set means scores are compressed. needs tuning with real corpus
- hinglish TTS falls back to english — pure hindi works fine, mixed script causes unnatural output. proper fix needs a normalization layer before the tts call
- no streaming retrieval on partial transcripts yet — current flow waits for full audio before transcribing. production version would debounce on partial whisper output to cut latency
- voice input tested on Chrome only
- `/debug` endpoint left in — remove before any real deployment
