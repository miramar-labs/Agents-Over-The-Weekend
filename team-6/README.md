# Team 6 AI Agent Project

## Project Description
This project aims to build a LangChain-powered intelligent agent that performs query-driven research and Q&A over YouTube videos, using their transcripts as the primary knowledge source.
The system uses a Retrieval-Augmented Generation (RAG) architecture and adds intelligent behaviors via LangChain tools and agents. It allows users to:
- Search for videos on a given topic
- Retrieve and process transcripts
- Embed the transcript into a vector database
- Query the content with a natural language question
- Optionally filter results by speaker, topic relevance, or semantic strength
- Provide detailed, grounded answers using OpenAI's GPT models

### Problem Statement
Most YouTube videos contain valuable discussions, interviews, or insights that are locked in unstructured, unlabeled transcripts. These transcripts:
- Lack speaker information
- Are not organized by topic
- Cannot be queried meaningfully without context
We aim to transform these raw transcripts into searchable, semantically rich sources of truth that an LLM can accurately reason over.

### Project Scope
#### Phase 1: Core RAG Workflow
- Search YouTube using the YouTube Data API
- Retrieve English transcripts using youtube-transcript-api
- Split transcripts into text chunks
- Embed chunks using sentence-transformers
- Store them in FAISS (or Chroma)
- Implement a LangChain RetrievalQA chain for basic question answering
##### Phase 2: LangChain Agent with Modular Tools
- Wrap each action (search, fetch, embed, retrieve) as a LangChain Tool
- Build a ReAct-style agent that selects tools dynamically
- Optionally orchestrate with LangGraph for more controlled flow
- Add logging, source tracking, and memory
#### Phase 3: Speaker-Aware Reasoning (Optional Enhancement)
- Use an LLM to segment transcripts by inferred speakers
- Tag each chunk with speaker metadata (e.g., "Host", "Witten")
- Enable speaker-specific filtering or biasing during retrieval
- Support user queries like: “What did Witten say about Lie groups?”
#### Phase 4: Interface (Optional)
- CLI interface for queries
- Optional Streamlit or FastAPI UI
- Optionally deploy to Hugging Face Spaces or server

### Success Criteria
- User can ask a question like:
“What did the expert say about the symmetry in Lie algebras?”
And receive a grounded, specific answer with optional speaker attribution.
- The system correctly filters out irrelevant videos/transcripts.
- The system can scale to new queries and topics with minimal rework.

### Design Plan — LangChain Agent + Graph Flow
#### Step 1: Define System Roles as LangChain Tools or Nodes
| Function |	LangChain Component	| Description |
|----------|----------------------|-------------|
|YouTube Video Search	|Tool / Function node|	Takes user query → searches via YouTube API|
|Video Relevance Check|	Tool / Evaluator	|Ranks based on title + description + embedding |similarity|
|Transcript Retriever|	Tool / Function|	Gets transcripts using youtube-transcript-api|
|Chunk + Embed Text|	Built-in RAG helper|	Breaks into chunks, embeds with SentenceTransformer|
|Vector Store Search|	Retriever Tool|	FAISS or Chroma retriever, possibly hybrid|
|Query Re-Ranker / Booster	|Node or Custom Tool	|Optional reranker using OpenAI scoring or BM25 fallback|
|Answer Generation	|OpenAI LLM|	Sends best matched content for detailed summary|
|Agent Orchestrator	|LangGraph / LangChain Agent|	Controls the above flow and decisions|


## Dev Notes
#### Docs
- [LangChain](https://python.langchain.com/docs/introduction/)
- [LangChain Python API](https://python.langchain.com/api_reference/)
#### API Keys
- [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
  - get API key `GOOGLE_API_KEY`

- [LangSmith](https://smith.langchain.com/)
  - get API key `LANGCHAIN_API_KEY`

- [OpenAI Platform](https://platform.openai.com/docs/overview)
  - get API key`OPENAI_API_KEY`
### Linux/WSL:
  
- First set up a python venv and work in that:

        # (*if* we need sqlite3) build a version of python with sqlite3 built-in 
        sudo apt-get update
        sudo apt-get install libsqlite3-dev

        # build some version of python
        pyenv install 3.13.0

        # now create a virtual env with this sql-enabled python
        pyenv virtualenv 3.13.0 aotw

        # set it as the venv for team-6 folder
        cd Agents-Over-The-Weekend/team-6

        pyenv local aotw
        
        # ensure sqlite3 not listed in requirements.txt
        
        # install any packages we need
        pip install -r requirements.txt

- Create a `.env` file in the team-6 folder to store your ENV vars/API keys:

        LANGCHAIN_API_KEY=
        LANGCHAIN_TRACING_V2=true
        GOOGLE_API_KEY=
