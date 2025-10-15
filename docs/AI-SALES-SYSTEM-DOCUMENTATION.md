# AI-Sales AI-System Documentation

## Overview

### What is AI-Sales?

AI-Sales is an intelligent document processing and question-answering system that turns your documents into a searchable knowledge base. Instead of manually searching through PDFs and videos, users can simply ask questions in natural language and get accurate answers instantly.

**How It Works:**
The system uses AI to read and understand your documents, breaks them into meaningful chunks, and stores them in a way that makes finding relevant information fast and accurate. When you ask a question, it searches for the most relevant information and generates a natural answer using only your uploaded content.

**Three Main Operations:**
1. **Add Documents to Knowledge Base** - Upload PDFs or videos, system extracts and processes content
2. **Remove Documents from Knowledge Base** - Delete unwanted documents from the system
3. **Ask Questions** - Get accurate answers by retrieving similar information from knowledge base

---

## 1. Adding Documents to Knowledge Base

### Process Flow

**Step 1: Document Download**
- Document is stored in AWS S3
- System downloads the document from S3 URL
- Detects file type (PDF or video)

**Step 2: Content Extraction**

For PDFs:
- PyMuPDF tool opens and reads the PDF
- Extracts text page by page
- Preserves document structure and page markers

For Videos:
- FFmpeg tool extracts audio track from video
- Converts audio to WAV format (optimized for speech recognition)
- AssemblyAI service transcribes the audio to text
- Includes language detection and punctuation

**Step 3: Semantic Chunking (AI Intelligence)**

This is where AI intelligence creates meaningful chunks:

**How Semantic Chunking Works:**
1. Text is split into paragraphs (natural document structure)
2. OpenAI generates embeddings (vector representations) for each paragraph
3. System calculates similarity between consecutive paragraphs
4. Identifies topic boundaries where similarity drops significantly
5. Groups related paragraphs together into coherent chunks

**Example:**
```
Paragraph 1: "Introduction to AI" → 
Paragraph 2: "AI applications" → High similarity
Paragraph 3: "AI benefits" → High similarity
[CHUNK 1 created from paragraphs 1-3]

Paragraph 4: "Privacy concerns" → Low similarity (topic change!)
Paragraph 5: "Data protection laws" → High similarity
[CHUNK 2 created from paragraphs 4-5]
```

**Why This Matters:**
- Chunks contain complete topics (not randomly split)
- Related information stays together
- Better retrieval when answering questions

**Chunk Optimization:**
- Small chunks (< 100 words) are merged with neighbors
- Large chunks (> 1500 words) are split at paragraph boundaries
- Each chunk is 100-1500 words with complete context

**Step 4: Vector Embedding Generation**
- OpenAI converts each chunk into a 1536-dimensional vector
- Vectors capture semantic meaning (similar concepts have similar vectors)
- Processes chunks in batches for efficiency

**Step 5: Storage in Vector Database**
- Vectors stored in Pinecone (specialized vector database)
- Separate indexes for PDFs and videos
- Each document gets its own namespace for isolation
- Metadata stored with each vector:
  - Original text content
  - Filename and file type
  - Chunk position and index
  - S3 URL reference

**Step 6: Status Update**
- System updates backend with "COMPLETED" status
- Document is now searchable in knowledge base

---

## 2. Removing Documents from Knowledge Base

### Process Flow

**Step 1: Removal Request**
- System receives request with document S3 URL
- Extracts filename from S3 URL
- Updates backend status to "PROCESSING"

**Step 2: Locate Document in Pinecone**
- Identifies correct index (PDF or video)
- Finds document's namespace (each document has unique namespace)
- Retrieves statistics about vectors to be deleted

**Step 3: Delete Vectors**

Primary method:
- Deletes entire namespace (all chunks for that document)
- Fast and efficient removal

Fallback method (if namespace deletion fails):
- Searches for all vectors matching the S3 URL
- Collects all vector IDs
- Deletes vectors by ID list

**Step 4: Status Update**
- Updates backend status to "PENDING" (removed from knowledge base)
- Returns count of chunks removed
- Document no longer searchable

---

## 3. Answering Questions (RAG Intelligence)

### Process Flow

**Step 1: User Asks Question**

User submits a question to the system:
- Example: "What is Article 21?"
- Can be first question (new conversation)
- Can be follow-up question (continuing conversation)

System receives:
- The question text
- Conversation ID (if continuing existing conversation)
- Previous conversation history (past questions and answers)

**Step 2: Query Enhancement**

Before searching, system checks if the question needs context:

**Analyzing the Question:**
- Reviews conversation history (if available)
- Checks if this is a follow-up question or standalone question
- Looks for follow-up indicators like "why not", "tell me more", "explain that"

**Enhancement Example:**
```
User's first question: "Can I kill someone according to constitution?"
System answers: "No, you cannot kill someone. The constitution protects right to life..."

User's follow-up: "Why not?"
Problem: "Why not?" is unclear without context

System enhancement:
- Sees previous discussion was about killing someone and constitution
- Rephrases to: "Why can't I kill someone according to constitutional rights?"
- Now the question is clear and searchable
```

**Standalone Questions Stay Unchanged:**
- "What is the constitution?" → No rephrasing needed
- "Tell me about Maharashtra" → Clear as-is, no enhancement

**Step 3: Convert Question to Vector**

System converts the enhanced question into numerical form:
- Uses OpenAI to create vector embedding (1536 numbers)
- This vector represents the meaning of the question
- Similar questions will have similar vectors

**Step 4: Search Knowledge Base**

System searches for similar information in Pinecone:

1. Takes the question vector
2. Searches across PDF index (finds relevant PDF chunks)
3. Searches across video index (finds relevant video transcript chunks)
4. Each chunk's vector is compared with question vector
5. Calculates similarity score (how relevant each chunk is)
6. Collects all matching chunks from all documents
7. Sorts by similarity score (most relevant first)
8. Selects top 3 most relevant chunks

**Why Vector Search Works:**
- Matches by meaning, not exact words
- Question: "automobile" → Finds chunks about "cars"
- Works across languages
- Understands synonyms and related concepts

**Step 5: Classify Query Intent**

System determines what type of question this is:

**Two Types:**

**Type A: General Conversation**
- Greetings: "hello", "hi", "thanks", "good morning"
- Personal statements: "my name is John", "I am from Delhi"
- Memory questions: "what did I ask earlier?", "what was my first question?"
- Confirmations: "yes", "okay", "sure"

**Type B: Knowledge Question**
- Factual questions: "what is...", "explain...", "how does...", "when was..."
- Questions requiring information from uploaded documents
- Follow-ups to knowledge discussions

**Context-Aware Classification:**
- "Tell me more" after greeting → Type A (General)
- "Tell me more" after knowledge answer → Type B (Knowledge)

**Step 6: Generate Response**

Based on classification, system takes different paths:

**Path A: General Conversation Response**

For greetings/personal questions:
- Responds warmly and briefly
- Example: "Hello! How can I help you today?"

For memory questions:
- Recalls information from conversation history
- Example: "You asked about killing someone according to constitution in your first question"

For out-of-scope questions:
- Sets clear boundaries politely
- Example: "I can only answer using your uploaded documents. Please upload relevant materials if you'd like help with this topic."

**Path B: Knowledge Question Response (RAG Process)**

This is where RAG happens:

1. **Combine Retrieved Information + Question**
   - Takes the 3 most relevant chunks found earlier
   - Adds user's question
   - Prepares instruction for AI

2. **Send to OpenAI with Rules**
   - Use ONLY information from the 3 retrieved chunks
   - Do NOT add any external knowledge
   - Respond in same language as user's question (English/Hindi/Hinglish)
   - If chunks don't fully answer, say what's available and what's missing
   - Write naturally (don't mention "chunks" or "documents")

3. **OpenAI Generates Answer**
   - Reads the retrieved chunks
   - Understands the question
   - Creates natural answer using only provided information
   
**Example Flow:**
```
Question: "What is Article 21?"
Retrieved Chunks: 
  - Chunk 1: "Article 21 of Indian Constitution states..."
  - Chunk 2: "Right to life and personal liberty..."
  - Chunk 3: "Supreme Court interpretations of Article 21..."

OpenAI generates: "Article 21 guarantees the right to life and personal liberty. It is one of the fundamental rights in the Indian Constitution..."
```

4. **Return Answer to User**
   - User receives natural, conversational answer
   - Answer is based only on their uploaded documents
   - Source information included (which chunks were used)

**When No Relevant Information Found:**
- System clearly states: "I don't have information about that topic in my knowledge base"
- Suggests: "Please upload relevant documents to help me answer this question"
- Does NOT make up information

**Step 7: Maintain Conversation Continuity**

After providing the answer:

**Save Conversation State:**
- System generates unique conversation ID (from OpenAI)
- This ID tracks the conversation thread
- Stores question and answer in conversation history

**For First Message:**
- Creates conversation title automatically
- Example: Question "What is AI?" → Title: "AI Discussion"
- Example: Question "Hello" → Title: "General Chat"

**For Follow-up Messages:**
- Uses existing conversation ID
- Maintains full context automatically
- No need for user to repeat information

**Multi-Turn Example:**
```
Turn 1: "What is the constitution?"
Answer: [Explains constitution]
Conversation ID: conv_123

Turn 2: "When was it created?" 
System knows "it" = constitution (from conversation history)
Uses conversation ID: conv_123
Answer: [Explains creation date]

Turn 3: "Who wrote it?"
System maintains context
Answer: [Explains authors]

Turn 4: "What are its main principles?"
System continues understanding "its" = constitution
Answer: [Explains principles]
```

**Result:**
- Natural, flowing conversation
- System remembers context
- User doesn't need to repeat information
- Questions can reference previous answers

---

## Tools & Technologies Used

**Document Processing:**
- **PyMuPDF**: Extracts text from PDF files
- **FFmpeg**: Extracts audio from videos
- **AssemblyAI**: Transcribes audio to text

**AI Intelligence:**
- **OpenAI GPT-4o**: Generates conversational responses
- **OpenAI Embeddings**: Converts text to vectors (text-embedding-3-small model)
- **Semantic Analysis**: Paragraph-level similarity calculation

**Storage:**
- **AWS S3**: Stores original document files
- **Pinecone**: Vector database for semantic search

**Process Management:**
- Status tracking with backend system
- Error handling and retry logic
- Health monitoring of all services

---

## Key Benefits

**For Users:**
- Upload documents and get instant searchable knowledge base
- Ask questions in natural language (English, Hindi, Hinglish, or any language)
- Get accurate answers based only on uploaded documents
- Continue conversations naturally with context awareness
- No need to repeat information in follow-up questions

**Intelligence Features:**
- Understands follow-up questions automatically (knows what "it" or "that" refers to)
- Finds relevant information by meaning, not just exact keywords
- Preserves document context through smart topic-based chunking
- Handles both PDFs and videos seamlessly
- Multi-language support with automatic language detection

**Reliability & Trust:**
- Clear boundaries - only answers from your knowledge base
- No made-up information (no hallucination)
- Source attribution included with answers
- When information is not available, system clearly states it
- Suggests uploading relevant documents when needed

---

![Add Document Flow](data:image/png;base64,
