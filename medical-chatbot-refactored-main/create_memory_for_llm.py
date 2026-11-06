from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data):
    print("📄 Loading PDFs from:", data)
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print("✅ Loaded", len(documents), "pages from PDF(s)")
    return documents

def create_chunks(extracted_data):
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(extracted_data)
    print("✅ Created", len(chunks), "chunks")
    return chunks

def get_embedding_model():
    print("🧠 Loading embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index():
    print("⚡ Starting FAISS index build...")

    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    embedding_model = get_embedding_model()

    print("🔢 Generating embeddings and building FAISS index...")
    db = FAISS.from_documents(chunks, embedding_model)

    print("💾 Saving FAISS index to:", DB_FAISS_PATH)
    db.save_local(DB_FAISS_PATH)

    print("🎉 DONE! FAISS index successfully created.")

if __name__ == "__main__":
    build_faiss_index()
