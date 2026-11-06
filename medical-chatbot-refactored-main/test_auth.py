import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

groq_key = os.environ.get("GROQ_API_KEY", "").strip()

print("=" * 50)
print("🔍 Checking Groq API Configuration...")
print("=" * 50)

if not groq_key or groq_key.startswith("your_"):
    print("❌ ERROR: GROQ_API_KEY is not configured or still contains the placeholder.")
    print("👉 Please open .env and put your real Groq API key: GROQ_API_KEY=gsk_...")
    exit(1)

# Mask key for privacy
masked = groq_key[:6] + "..." + groq_key[-4:] if len(groq_key) > 10 else "***"
print(f"🔑 Key detected: {masked}")

try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)
    print("📡 Sending test ping to Groq API (model: llama-3.1-8b-instant)...")
    res = llm.invoke("Hi, respond with 'Authentication successful' and nothing else.")
    print("✅ SUCCESS! Groq response received:")
    print("🤖", res.content.strip())
except Exception as e:
    print("❌ Groq Authentication Failed:")
    print(str(e))
    exit(1)

