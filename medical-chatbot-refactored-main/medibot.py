import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    st.title("Ask Chatbot!")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        saved_key = os.environ.get("GROQ_API_KEY", "")
        if saved_key.startswith("your_"):
            saved_key = ""
        api_key_input = st.text_input(
            "Groq API Key",
            value=saved_key,
            type="password",
            help="Get your free key at https://console.groq.com/keys"
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input.strip()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
                
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            groq_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not groq_key or groq_key.startswith("your_"):
                st.warning("⚠️ **Groq API Key Required**: Please enter your Groq API key in the sidebar or in `.env` to enable AI responses. [Get a free key here](https://console.groq.com/keys).")
                return

            GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # Change to any supported Groq model
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=groq_key,
            )
            
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

            # Document combiner chain (stuff documents into prompt)
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

            # Retrieval chain (retriever + doc combiner)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 5}), combine_docs_chain)

            response=rag_chain.invoke({'input': prompt})

            result=response["answer"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            err_msg = str(e)
            if "invalid_api_key" in err_msg or "401" in err_msg:
                st.error("🔑 Authentication Error (401): Invalid Groq API Key. Please verify your `GROQ_API_KEY` in `.env` from https://console.groq.com/keys.")
            else:
                st.error(f"Error: {err_msg}")

if __name__ == "__main__":
    main()