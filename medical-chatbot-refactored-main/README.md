# 🩺 AI Medical Chatbot (RAG Clinical Assistant)

An intelligent clinical reference and medical question-answering assistant built with **LangChain**, **FAISS Vector Store**, **Sentence Transformers**, **Groq LLMs (`llama-3.1-8b-instant`)**, and **Streamlit**.

The chatbot uses Retrieval-Augmented Generation (RAG) to ground answers in verified medical literature and clinical guides, preventing hallucinations and citing supporting medical knowledge passages.

---

## 🧠 System Architecture

```text
  [Medical PDFs / Textbooks]
              │
              ▼
   [PyPDFLoader / DirectoryLoader]
              │
              ▼
  [RecursiveCharacterTextSplitter] (chunk_size=1800, overlap=300)
              │
              ▼
  [Sentence-Transformers Embeddings] (all-MiniLM-L6-v2)
              │
              ▼
     [FAISS Vector Store] (vectorstore/db_faiss)
              │
User Query ───┴──► [FAISS Retriever (top-k)]
                         │
                         ▼
             [LangChain Prompt Assembly]
                         │
                         ▼
        [Groq LLM (llama-3.1-8b-instant)]
                         │
                         ▼
               [Answer + Context Sources]
```

---

## ✨ Features

- **Semantic Vector Retrieval**: Fast cosine-similarity search powered by FAISS and 384-dimensional embeddings.
- **Ultra-Fast LLM Inference**: Powered by Groq's LPU architecture running `llama-3.1-8b-instant`.
- **Dual Configuration Modes**: Provide API keys securely via `.env` or interactively in the Streamlit UI sidebar.
- **Traceable Responses**: Returns grounded answers with source document attribution.
- **Safe Error Handling**: Proactive validation prevents 401 authentication failures and masks credentials in logs.
- **CLI & Web Interfaces**: Run the Streamlit web chat application or use the terminal RAG pipeline.

---

## 📁 Project Structure

```text
Ai-Medical-Chat-Box/
├── .env.example                         # Safe environment template
├── .gitignore                           # Git ignore rules for virtualenv, .env, and caches
├── README.md                            # Project documentation and setup guide
└── medical-chatbot-refactored-main/
    ├── .env                             # Local secrets (never committed to Git)
    ├── .env.example                     # Reference environment variables
    ├── connect_memory_with_llm.py       # CLI interface for RAG querying
    ├── create_memory_for_llm.py         # Script to index medical PDFs into FAISS
    ├── data/                            # Storage directory for source medical PDFs
    ├── medibot.py                       # Main Streamlit web application
    ├── pyproject.toml                   # Project metadata and dependencies
    ├── requirements.txt                 # Pinned dependencies
    ├── test_auth.py                     # Independent API authentication test tool
    └── vectorstore/
        └── db_faiss/                    # Persisted FAISS index files
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.10, 3.11, or 3.12 (64-bit recommended)
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/rahulsamal12/Ai-Medical-Chat-Box.git
cd Ai-Medical-Chat-Box/medical-chatbot-refactored-main
```

### 3. Create & Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your credentials:
   ```env
   # Groq API Key (Required for medibot.py and connect_memory_with_llm.py)
   # Obtain a free key at https://console.groq.com/keys
   GROQ_API_KEY=xxxxxxxxxxxxxxxxxxxxx

   # HuggingFace Token (Optional: used only if switching to HuggingFace Inference API)
   # Obtain at https://huggingface.co/settings/tokens
   HF_TOKEN=xxxxxxxxxxxxxxxxxxxxx

   # Suppress unauthenticated tracing telemetry warnings
   LANGCHAIN_TRACING_V2=false
   ```

> [!NOTE]
> Never commit `.env` containing live secrets. The repository is configured to exclude all `.env` files automatically.

---

## 🗂️ Building the Medical Memory (Vector Store)

If you have custom medical books or documents:
1. Place your PDF files inside the `data/` directory (e.g. `medical-chatbot-refactored-main/data/`).
2. Run the indexing script:
   ```bash
   python create_memory_for_llm.py
   ```
This extracts text, splits it into semantic chunks, creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and persists the index under `vectorstore/db_faiss/`.

---

## 🚀 Running the Application

### Option A: Streamlit Web Chatbot (Recommended)

Start the interactive web app:
```bash
streamlit run medibot.py
```
Or with exact path in PowerShell:
```powershell
.\.venv\Scripts\streamlit.exe run medibot.py --server.port 8501
```

Open your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

You can also enter or update your Groq API Key directly in the sidebar **⚙️ Configuration** input field.

### Option B: CLI Terminal Interface
```bash
python connect_memory_with_llm.py
```
Enter a medical query in the prompt, such as:
```text
Write Query Here: How is hypertension managed?
```

### Option C: Authentication & Connectivity Diagnostic
```bash
python test_auth.py
```
This utility validates your `.env` loading, masks credentials, checks connectivity to Groq, and confirms that LLM completions work without starting the web UI.

---

## 💬 Example Questions to Try

- *"What is hypertension, and what are its standard management guidelines?"*
- *"What are the classic symptoms of Type 2 diabetes mellitus?"*
- *"How is acute asthma managed, and what triggers should patients avoid?"*
- *"What is the standard supportive treatment for viral upper respiratory infections?"*

---

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Error code: 401 (`invalid_api_key`)** | `GROQ_API_KEY` is missing or contains the placeholder | Obtain a key at [console.groq.com/keys](https://console.groq.com/keys) and enter it in `.env` or the sidebar. |
| **`FAISS.load_local ... file not found`** | Vector store has not been generated yet | Run `python create_memory_for_llm.py` with PDFs in `data/`. |
| **`LangSmithMissingAPIKeyWarning`** | LangChain attempting telemetry | Add `LANGCHAIN_TRACING_V2=false` to `.env`. |
| **PowerShell script execution policy error** | Execution policy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating the virtualenv. |

---

## ⚖️ Medical Disclaimer

> [!WARNING]
> This application is strictly intended for educational, academic, and informational reference purposes. It does **not** constitute medical advice, diagnosis, or clinical treatment plans. Always consult a qualified licensed healthcare provider for medical conditions, prescriptions, or urgent health concerns.

---

## 📄 License
This project is open-source under the MIT License.
