# RAG pipeline implementation using LangChain, Chroma, and Gemini 

import os
from typing import List
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import streamlit as st # For potential secrets management integration

# --- Constants ---
CHROMA_PERSIST_DIR = os.path.join("embeddings", "chroma_store")
# Ensure the directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Try loading API Key from .env first
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# If not found in .env, try Streamlit secrets (for deployment)
# if not GOOGLE_API_KEY:
#     try:
#         GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
#     except KeyError:
#          # Handle the case where the key is not found in either place
#          # You might want to raise an error or use a default behavior
#          print("Warning: GOOGLE_API_KEY not found in .env or Streamlit secrets.")
#          GOOGLE_API_KEY = None # Or raise an error

# --- Embedding Model ---
def get_embedding_model():
    """Initializes and returns the Google Generative AI Embedding model."""
    if not GOOGLE_API_KEY:
         raise ValueError("Google API Key is required for embeddings.")
    # Specify the model explicitly, e.g., "models/embedding-001"
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# --- Vector Store Functions ---
vectorstore = None # Global variable to hold the vector store instance

def create_vectorstore(chunks: List[Document], force_recreate: bool = False):
    """Creates or loads a Chroma vector store from document chunks."""
    global vectorstore
    embedding_model = get_embedding_model()

    if os.path.exists(CHROMA_PERSIST_DIR) and not force_recreate:
        print(f"Loading existing vector store from: {CHROMA_PERSIST_DIR}")
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
    else:
        print(f"Creating new vector store in: {CHROMA_PERSIST_DIR}")
        if not chunks:
            raise ValueError("Cannot create vector store with empty document chunks.")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PERSIST_DIR
        )
    print("Vector store created/loaded successfully.")
    return vectorstore

def get_retriever(k: int = 5):
    """Gets the retriever from the initialized vector store."""
    global vectorstore
    if vectorstore is None:
        # Attempt to load if not initialized (e.g., after app restart)
        try:
             vectorstore = create_vectorstore([], force_recreate=False) # Try loading
        except Exception as e:
             raise RuntimeError(f"Vector store not initialized and failed to load: {e}. Please process a PDF first.")

    if vectorstore is None:
        raise RuntimeError("Vector store is still not available after attempting to load.")

    return vectorstore.as_retriever(search_kwargs={'k': k})

# --- Language Model (LLM) ---
def get_llm():
    """Initializes and returns the Google Generative AI LLM."""
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key is required for the LLM.")
    # Choose appropriate Gemini model, e.g., "gemini-1.5-flash-latest"
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.7)

# --- RAG Chain Creation ---

# Prompt for RAG (using context from PDF)
RAG_QUESTION_PROMPT_TEMPLATE = """
**Context:**
{context}

**Görevin:** Yukarıdaki bağlamı kullanarak, belirtilen konuyla ilgili **{num_questions} adet**, **{level}** zorluk seviyesinde, çoktan seçmeli (4 seçenekli: A, B, C, D) test sorusu hazırla.

*   Eğer konu **"Belgenin Geneli"** olarak belirtilmişse, soruları bağlamın genelini kapsayacak şekilde oluştur.
*   Eğer **spesifik bir konu** belirtilmişse ({topic}), soruları o konuyla doğrudan ilgili olarak oluştur.

Sorular bağlamdaki bilgilere dayanmalıdır.

**İstenen Çıktı Formatı:**
Soruları, seçenekleri, cevap anahtarını ve **her sorunun türetildiği kaynak bilgisini (bağlamdaki 'Kaynak:' ile başlayan satır)** aşağıdaki JSON formatında, **kesinlikle başka hiçbir ek metin olmadan** ver:

```json
{{
  "sorular": [
    {{
      "soru_no": 1,
      "soru_metni": "[Soru metni buraya gelecek]",
      "secenekler": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
      "kaynak": "[Sorunun türetildiği kaynak metni]"
    }}
    // ... more questions
  ],
  "cevap_anahtari": {{ "1": "A", "2": "C", ... }}
}}
```

**Önemli Notlar:**
*   Sadece ve sadece istenen JSON formatında çıktı ver.
*   JSON'un geçerli olduğundan emin ol.
*   Soruların cevapları kesinlikle verilen bağlamda bulunmalı.
*   Seçenekler (A, B, C, D) net ve anlaşılır olmalı.
*   Sorular, belirtilen **{topic}** (veya genel kapsam) ile ilgili ve **{level}** zorluk seviyesine uygun olmalı.
*   Her soru için **'kaynak'** bilgisini eklemeyi unutma.

JSON Çıktısı:
"""

RAG_QUESTION_PROMPT = PromptTemplate(
    input_variables=["context", "topic", "num_questions", "level"],
    template=RAG_QUESTION_PROMPT_TEMPLATE,
)

# Prompt for Topic-Only Generation (no context)
TOPIC_ONLY_PROMPT_TEMPLATE = """
**Görevin:** Genel bilgini kullanarak, **{topic}** konusuyla ilgili **{num_questions} adet**, **{level}** zorluk seviyesinde, çoktan seçmeli (4 seçenekli: A, B, C, D) test sorusu hazırla.

**İstenen Çıktı Formatı:**
Soruları, seçenekleri ve cevap anahtarını aşağıdaki JSON formatında, **kesinlikle başka hiçbir ek metin olmadan** ver:

```json
{{
  "sorular": [
    {{
      "soru_no": 1,
      "soru_metni": "[Soru metni buraya gelecek]",
      "secenekler": {{ "A": "...", "B": "...", "C": "...", "D": "..." }}
      // Kaynak alanı olmayacak
    }}
    // ... more questions
  ],
  "cevap_anahtari": {{ "1": "A", "2": "C", ... }}
}}
```

**Önemli Notlar:**
*   Sadece ve sadece istenen JSON formatında çıktı ver.
*   JSON'un geçerli olduğundan emin ol.
*   Sorular doğrudan **{topic}** ile ilgili ve **{level}** zorluk seviyesine uygun olmalı.
*   Seçenekler (A, B, C, D) net ve anlaşılır olmalı.

JSON Çıktısı:
"""

TOPIC_ONLY_PROMPT = PromptTemplate(
    input_variables=["topic", "num_questions", "level"],
    template=TOPIC_ONLY_PROMPT_TEMPLATE,
)

def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string."""
    # Use the 'source' metadata prepared in utils.py
    return "\n\n".join(f"Kaynak: {doc.metadata.get('source', 'Bilinmiyor')}\n{doc.page_content}" for doc in docs)

def create_rag_chain(retriever, llm):
    """Creates the RAG chain (retrieval + generation)."""
    def retrieve_context(input_dict):
        retrieved_docs = retriever.invoke(input_dict["topic"])
        return format_docs(retrieved_docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=retrieve_context
          )
        | RAG_QUESTION_PROMPT # Use the RAG prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def create_topic_only_chain(llm):
    """Creates a chain that generates questions based only on the topic."""
    topic_chain = (
        TOPIC_ONLY_PROMPT # Use the topic-only prompt
        | llm
        | StrOutputParser()
    )
    return topic_chain

# Example usage would differentiate:
# rag_chain.invoke({"topic": ..., "num_questions": ..., "level": ...}) # Requires processed PDF
# topic_chain = create_topic_only_chain(get_llm())
# topic_chain.invoke({"topic": ..., "num_questions": ..., "level": ...}) # Doesn't need PDF

# Example usage (for testing purposes):
# if __name__ == '__main__':
#     # This part requires a PDF to be processed and Chroma DB populated first.
#     # You would typically call this from main.py
#     try:
#         # 1. Ensure you have processed a PDF and have chunks
#         # dummy_chunks = [Document(page_content="Python is a programming language.", metadata={"source": "test.pdf"})]
#         # create_vectorstore(dummy_chunks, force_recreate=True)

#         # 2. Get retriever and LLM
#         retriever = get_retriever()
#         llm = get_llm()

#         # 3. Create the chain
#         rag_chain = create_rag_chain(retriever, llm)

#         # 4. Invoke the chain
#         topic_to_test = "Python basics"
#         num_q = 2
#         response = rag_chain.invoke({"topic": topic_to_test, "num_questions": num_q})
#         print("Generated Exam:")
#         print(response)

#     except ValueError as ve:
#          print(f"Configuration Error: {ve}")
#     except RuntimeError as re:
#          print(f"Runtime Error: {re}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}") 