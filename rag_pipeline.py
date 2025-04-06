# RAG pipeline implementation using LangChain, Chroma, and Gemini 

import os
import logging
import shutil # Import shutil for directory operations
import gc   # Import garbage collector
import time # Import time for sleep
import re # Import regex for sanitization
from typing import List, Optional
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
# import streamlit as st # Keep commented unless actively using secrets this way

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
CHROMA_BASE_DIR = os.path.join("embeddings", "chroma_db") # Base directory for all stores
EMBEDDING_MODEL_NAME = "models/embedding-001"
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Or another preferred model
DEFAULT_RETRIEVER_K = 5 # Number of documents to retrieve
DEFAULT_LLM_TEMPERATURE = 0.7
MAX_DELETE_RETRIES = 3
INITIAL_DELETE_DELAY = 0.5 # seconds

# Ensure the base Chroma directory exists
os.makedirs(CHROMA_BASE_DIR, exist_ok=True)

# --- Helper Function ---
def sanitize_store_id(filename: str) -> str:
    """Removes or replaces characters invalid for directory names."""
    if not filename:
        return "default_store"
    # Remove leading/trailing whitespace
    sanitized = filename.strip()
    # Replace invalid characters (including path separators) with underscore
    sanitized = re.sub(r'[<>:"/\\|?*.,;=\[\]]+', '_', sanitized)
    # Replace sequences of underscores with a single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length to avoid issues with long paths (optional)
    sanitized = sanitized[:100]
    # Use a default if sanitization results in empty string
    return sanitized if sanitized else "default_store"

# --- API Key Loading ---
def load_google_api_key():
    """Loads the Google API Key, prioritizing .env then Streamlit secrets."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Example: How you might integrate Streamlit secrets if needed
    # if not api_key:
    #     try:
    #         api_key = st.secrets["GOOGLE_API_KEY"]
    #         logging.info("Loaded Google API Key from Streamlit secrets.")
    #     except (AttributeError, KeyError):
    #         logging.warning("Streamlit secrets not available or key not found.")
    #         pass # Keep api_key as None or the value from .env

    if not api_key:
        logging.error("Google API Key not found in .env or Streamlit secrets.")
        # Raising ValueError signals a configuration problem that prevents proceeding.
        raise ValueError("Google API Anahtarı bulunamadı. Lütfen .env dosyasında veya Streamlit secrets içinde ayarlayın.")
    else:
        logging.info("Google API Key loaded successfully.")
    return api_key

GOOGLE_API_KEY = load_google_api_key() # Load on module import

# --- Embedding Model ---
def get_embedding_model():
    """Initializes and returns the Google Generative AI Embedding model."""
    # API key is checked during load_google_api_key()
    try:
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        raise RuntimeError("Embedding modeli başlatılamadı.") from e

# --- Vector Store Functions ---
# Removed global vectorstore variable

def _get_persist_directory(store_id: str) -> str:
    """Constructs the persistence directory path for a given store ID (UUID)."""
    # Basic validation for store_id format (optional)
    if not store_id or len(store_id) < 10: # Simple check, not full UUID validation
        raise ValueError(f"Invalid store_id provided: {store_id}")
    return os.path.join(CHROMA_BASE_DIR, store_id)

def create_vectorstore(chunks: List[Document], store_id: str, force_recreate: bool = False) -> Chroma:
    """Creates or loads a Chroma vector store for a specific store_id (UUID)."""
    embedding_model = get_embedding_model()
    persist_dir = _get_persist_directory(store_id)

    should_load = os.path.exists(persist_dir) and os.listdir(persist_dir) and not force_recreate

    loaded_store: Optional[Chroma] = None

    if should_load:
        logging.info(f"Attempting to load vector store for ID '{store_id}' from: {persist_dir}")
        try:
            loaded_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
            logging.info(f"Existing vector store for ID '{store_id}' loaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to load existing vector store for ID '{store_id}' from {persist_dir}: {e}. Attempting to recreate.", exc_info=True)
            force_recreate = True # Force recreation if loading failed
    else:
         force_recreate = True # Force recreate if dir doesn't exist, is empty, or force_recreate was True

    if force_recreate:
        logging.info(f"Recreating vector store for ID '{store_id}' in: {persist_dir}")
        gc.collect()

        if os.path.exists(persist_dir):
            retries = 0
            deleted = False
            while retries < MAX_DELETE_RETRIES:
                try:
                    if retries == 0: time.sleep(INITIAL_DELETE_DELAY)
                    logging.debug(f"Attempt {retries + 1}/{MAX_DELETE_RETRIES} to delete directory: {persist_dir}")
                    shutil.rmtree(persist_dir)
                    deleted = True
                    logging.info(f"Successfully deleted directory: {persist_dir}")
                    break
                except OSError as e:
                    logging.warning(f"Attempt {retries + 1} failed to delete directory {persist_dir}: {e}")
                    retries += 1
                    if retries < MAX_DELETE_RETRIES:
                        delay = INITIAL_DELETE_DELAY * (2 ** retries)
                        logging.info(f"Waiting {delay:.2f} seconds before retry...")
                        time.sleep(delay)
                        gc.collect()
                    else:
                        logging.error(f"Failed to delete directory {persist_dir} after {MAX_DELETE_RETRIES} attempts.", exc_info=True)
                        raise RuntimeError(f"Vektör deposu dizini (ID: {store_id}) silinemedi.") from e
            if deleted:
                os.makedirs(persist_dir, exist_ok=True)

        if not chunks:
            logging.error("Cannot create vector store with empty document chunks when recreation is required.")
            raise ValueError("Vektör deposu oluşturmak için belge parçacıkları (chunks) gerekli.")

        # Add detailed logging before the potentially failing call
        logging.info(f"Preparing to create Chroma store for ID '{store_id}' with {len(chunks)} chunks.")
        logging.debug(f"Target directory: {persist_dir}")
        if chunks:
            # Log a snippet of the first chunk for inspection
            try:
                 first_chunk_content_snippet = chunks[0].page_content[:200] # Log first 200 chars
                 first_chunk_metadata = chunks[0].metadata
                 logging.debug(f"First chunk content snippet: {first_chunk_content_snippet}...")
                 logging.debug(f"First chunk metadata: {first_chunk_metadata}")
            except Exception as log_e:
                 logging.warning(f"Could not log first chunk details: {log_e}")

        try:
            logging.info(f"Calling Chroma.from_documents for ID '{store_id}'...")
            loaded_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persist_dir
            )
            logging.info(f"New vector store for ID '{store_id}' created and persisted successfully.")
        except Exception as e:
            logging.error(f"Failed to create new vector store for ID '{store_id}': {e}", exc_info=True)
            raise RuntimeError(f"Yeni vektör deposu (ID: {store_id}) oluşturulamadı.") from e

    if loaded_store is None:
        logging.critical(f"Vector store for ID '{store_id}' is unexpectedly None after creation/loading attempt.")
        raise RuntimeError(f"Vektör deposu (ID: {store_id}) başlatılamadı.")

    return loaded_store

def get_retriever(store_id: str, k: int = DEFAULT_RETRIEVER_K):
    """Loads the vector store for the given store_id (UUID) and returns a retriever."""
    persist_dir = _get_persist_directory(store_id)
    logging.info(f"Getting retriever for store ID '{store_id}' from path: {persist_dir}")

    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        logging.error(f"Vector store directory for store_id '{store_id}' not found or is empty at {persist_dir}.")
        raise RuntimeError(f"ID '{store_id}' için vektör deposu bulunamadı. Lütfen ilgili PDF'i tekrar işleyin.")

    try:
        embedding_model = get_embedding_model()
        loaded_vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        logging.info(f"Successfully loaded vector store for ID '{store_id}' to create retriever.")
        return loaded_vectorstore.as_retriever(search_kwargs={'k': k})
    except Exception as e:
        logging.error(f"Failed to load vector store or create retriever for store_id '{store_id}' from {persist_dir}: {e}", exc_info=True)
        raise RuntimeError(f"Vektör deposu (ID: {store_id}) yüklenemedi veya retriever oluşturulamadı.") from e

# --- Language Model (LLM) ---
def get_llm():
    """Initializes and returns the Google Generative AI LLM."""
    # API key is checked during load_google_api_key()
    try:
        return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME,
                                    google_api_key=GOOGLE_API_KEY,
                                    temperature=DEFAULT_LLM_TEMPERATURE,
                                    convert_system_message_to_human=True) # Often helpful for Gemini
    except Exception as e:
        logging.error(f"Failed to initialize LLM model '{LLM_MODEL_NAME}': {e}", exc_info=True)
        raise RuntimeError("Dil modeli (LLM) başlatılamadı.") from e

# --- RAG Chain Creation ---

# Prompt Template for RAG (Context-based Question Generation)
# This template guides the LLM to generate multiple-choice questions based on
# the provided context retrieved from the vector store.
RAG_QUESTION_PROMPT_TEMPLATE = """
**Context:**
{context}

**Görevin:** Yukarıdaki bağlamı kullanarak, belirtilen konuyla ilgili **{num_questions} adet**, **{level}** zorluk seviyesinde, çoktan seçmeli (4 seçenekli: A, B, C, D) test sorusu hazırla.

*   Eğer konu **"Belgenin Geneli"** olarak belirtilmişse, soruları bağlamın genelini kapsayacak şekilde oluştur. ve sadece ders içeriği ile ilgili olan bölümlerden hazırla soruları.
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

# Prompt Template for Topic-Only Generation (No Context)
# This template guides the LLM to generate multiple-choice questions based on
# its general knowledge about the specified topic, without relying on retrieved context.
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
    """Helper function to format retrieved documents into a single string for context."""
    if not docs:
        return "Kullanılacak belge bağlamı bulunamadı."
    # Use the 'source' metadata prepared in utils.py for better traceability
    return "\n\n".join(f"Kaynak: {doc.metadata.get('source', 'Bilinmiyor')}\n{doc.page_content}" for doc in docs)

def create_rag_chain(retriever, llm):
    """Creates the RAG chain (retrieval + generation)."""

    # Use RunnableLambda for cleaner context retrieval step
    retrieve_context = RunnableLambda(
        lambda input_dict: format_docs(retriever.invoke(input_dict["topic"])))

    rag_chain = (
        RunnablePassthrough.assign(
            # Retrieve context based on the topic
            context=retrieve_context
        )
        # context=lambda input_dict: format_docs(retriever.invoke(input_dict["topic"])) # Alternative inline lambda
        # )
        | RAG_QUESTION_PROMPT # Format the prompt with context and other inputs
        | llm                 # Generate the response using the LLM
        | StrOutputParser()   # Parse the output as a string
    )
    logging.info("RAG chain created successfully.")
    return rag_chain

def create_topic_only_chain(llm):
    """Creates a chain that generates questions based only on the topic."""
    topic_chain = (
        TOPIC_ONLY_PROMPT # Format the prompt with topic, num_questions, level
        | llm             # Generate the response using the LLM
        | StrOutputParser()# Parse the output as a string
    )
    logging.info("Topic-only chain created successfully.")
    return topic_chain

# Example usage (keep commented out - intended for direct testing)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG) # Use DEBUG for detailed testing logs
#     try:
#         # --- Test RAG Chain ---
#         logging.info("--- Testing RAG Chain ---")
#         # 1. Create dummy data and vector store
#         dummy_chunks = [
#             Document(page_content="Python, nesne yönelimli, yorumlanan, modüler ve etkileşimli yüksek seviyeli bir dildir.",
#                      metadata={'source': 'python_doc.pdf - Sayfa 1, Parça 0'}),
#             Document(page_content="Guido van Rossum tarafından geliştirilmiştir. İlk sürümü 1991'de yayınlandı.",
#                      metadata={'source': 'python_doc.pdf - Sayfa 1, Parça 1'})
#         ]
#         create_vectorstore(dummy_chunks, force_recreate=True)

#         # 2. Get components
#         retriever = get_retriever(k=1)
#         llm = get_llm()

#         # 3. Create chain
#         rag_chain = create_rag_chain(retriever, llm)

#         # 4. Invoke
#         response_rag = rag_chain.invoke({"topic": "Python'un Tanımı", "num_questions": 1, "level": "Kolay"})
#         logging.info("RAG Chain Output:")
#         print(response_rag)

#         # --- Test Topic-Only Chain ---
#         logging.info("\n--- Testing Topic-Only Chain ---")
#         llm_topic = get_llm()
#         topic_chain = create_topic_only_chain(llm_topic)
#         response_topic = topic_chain.invoke({"topic": "Fransız İhtilali Nedenleri", "num_questions": 1, "level": "Orta"})
#         logging.info("Topic-Only Chain Output:")
#         print(response_topic)

#     except (ValueError, RuntimeError) as conf_err:
#          logging.error(f"Configuration or Runtime Error: {conf_err}", exc_info=True)
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during testing: {e}", exc_info=True) 