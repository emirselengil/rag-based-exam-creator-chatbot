# --- Patch for ChromaDB/SQLite on Streamlit Cloud ---
# This must be at the VERY top before any imports that might load sqlite3
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    # Optional: Add a check or log
    # import sqlite3
    # print(f"Using SQLite version: {sqlite3.sqlite_version}")
except ImportError:
    # Optional: Log if pysqlite3 is not found
    pass
# --- End Patch ---

import streamlit as st
import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import utils
import rag_pipeline

# --- Constants ---
# Session State Keys
S_PDF_PROCESSED = 'pdf_processed'
S_DOC_CHUNKS = 'doc_chunks'
S_VECTORSTORE_CREATED = 'vectorstore_created'
S_UPLOADED_FILENAME = 'uploaded_filename'
S_SESSION_ID = 'session_id'
S_EXAM_RESULTS = 'exam_results' # Stores raw response from LLM
S_EXAM_TO_DISPLAY = 'exam_to_display' # Stores full data of a selected historical exam
S_LAST_GENERATED_TOPIC = 'last_generated_topic'
S_LAST_SOURCE_PDF = 'last_source_pdf'
S_CURRENT_STORE_ID = 'current_store_id'
S_VECTORSTORE_READY_FOR_ID = 'vectorstore_ready_for_id'

# UI Texts & Placeholders
PAGE_TITLE = "Sınav Hazırlayan Chatbot"
PAGE_ICON = "🚀"
APP_TITLE = "🚀 Sınav Hazırlayan RAG Based Chatbot"
APP_DESC = """
Kendi ders notlarınızdan veya dokümanlarınızdan hızlıca çoktan seçmeli sınavlar oluşturun!

**Nasıl Kullanılır:**
1.  Sol menüden PDF dosyanızı yükleyin.
2.  Yine sol menüden sınavınız için istediğiniz ayarları (konu, soru sayısı vb.) belirtin.
3.  'Sınav Oluştur' butonuna tıklayın.
4.  Chatbot sizin için soruları hazırlayacaktır. (Cevap anahtarı ile birlikte!)
"""
SIDEBAR_HEADER = "Ayarlar"
PDF_UPLOAD_LABEL = "1. PDF Yükle"
EXAM_PARAMS_HEADER = "2. Sınav Özellikleri"
TOPIC_INPUT_LABEL = "Konu (PDF yüklemeden de sadece konuyla sınav oluşturabilirsiniz)"
TOPIC_INPUT_PLACEHOLDER = "Örn: Türkiye Coğrafyası"
LEVEL_SELECT_LABEL = "Seviye"
LEVEL_OPTIONS = ["Kolay", "Orta", "Zor"]
NUM_QUESTIONS_LABEL = "Soru Sayısı"
GENERATE_BUTTON_LABEL = "3. Sınav Oluştur"
HISTORY_HEADER = "Geçmiş Sınavlar"
VIEW_BUTTON_LABEL = "Görüntüle"
DELETE_BUTTON_LABEL = "Sil"
DOWNLOAD_HEADER = "💾 Sınavı İndir"
DOWNLOAD_PDF_LABEL = "📄 PDF olarak İndir"
DOWNLOAD_DOCX_LABEL = "📝 Word (DOCX) olarak İndir"

# Other Constants
MAX_QUESTIONS = 20
DEFAULT_QUESTIONS = 5
HISTORY_LIMIT = 15
DEFAULT_TOPIC_NAME = "Belgenin Geneli"
TOPIC_ONLY_DISPLAY_NAME = "Konu Bazlı"

# Define the target timezone (UTC+3)
TURKEY_TZ = timezone(timedelta(hours=3))

# --- Environment & Initialization ---

def load_api_key():
    """Loads Google API Key from .env or Streamlit secrets."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, AttributeError):
            st.error("Google API Anahtarı bulunamadı. Lütfen .env dosyasına veya Streamlit secrets'a ekleyin.")
            st.stop()
    return api_key

def initialize_database():
    """Initializes the database connection."""
    try:
        utils.init_db()
    except Exception as e:
        st.sidebar.error(f"Veritabanı başlatılamadı: {e}")
        st.stop()

def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if S_SESSION_ID not in st.session_state:
        st.session_state[S_SESSION_ID] = str(uuid.uuid4())
    defaults = {
        S_PDF_PROCESSED: False,
        S_DOC_CHUNKS: None,
        S_VECTORSTORE_CREATED: False,
        S_UPLOADED_FILENAME: None,
        S_EXAM_RESULTS: None,
        S_EXAM_TO_DISPLAY: None,
        S_LAST_GENERATED_TOPIC: None,
        S_LAST_SOURCE_PDF: None,
        S_CURRENT_STORE_ID: None,
        S_VECTORSTORE_READY_FOR_ID: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Helper Functions ---

def reset_exam_state():
    """Resets session state related to PDF processing and exam results."""
    st.session_state[S_PDF_PROCESSED] = False
    st.session_state[S_VECTORSTORE_CREATED] = False
    st.session_state[S_DOC_CHUNKS] = None
    st.session_state[S_EXAM_RESULTS] = None
    st.session_state[S_UPLOADED_FILENAME] = None
    st.session_state[S_CURRENT_STORE_ID] = None
    # Keep S_VECTORSTORE_READY_FOR_ID to know the last successful one

def parse_exam_json(json_str):
    """Parses the JSON string response from the LLM into a Python dict."""
    try:
        # Try cleaning potential markdown code fences first
        cleaned_response = json_str.strip().removeprefix("```json").removesuffix("```").strip()
        results = json.loads(cleaned_response)
        if not isinstance(results, dict) or "sorular" not in results or "cevap_anahtari" not in results:
            st.error("Oluşturulan sınav formatı anlaşılamadı (geçersiz yapı). Ham veri aşağıdadır:")
            st.code(json_str, language='json')
            return None
        return results
    except json.JSONDecodeError:
        st.error("Üretilen yanıt JSON formatında değil. Ham yanıt aşağıdadır:")
        st.code(json_str, language='text')
        return None
    except Exception as e:
        st.error(f"Sınav verisi ayrıştırılırken beklenmedik bir hata oluştu: {e}")
        st.code(json_str, language='text')
        return None

def display_parsed_exam(exam_data):
    """Displays the questions and answers from parsed exam data."""
    st.subheader("📝 Oluşturulan Sınav")
    for i, q in enumerate(exam_data.get("sorular", [])):
        st.markdown(f"**Soru {q.get('soru_no', i+1)}:** {q.get('soru_metni', 'Soru metni bulunamadı')}")
        kaynak = q.get('kaynak', None)
        if kaynak:
            st.caption(f"Kaynak: {kaynak}")
        options = q.get("secenekler", {})
        # Use a unique key for display
        st.radio("Seçenekler:",
                 options=[f"{key}: {value}" for key, value in options.items()],
                 key=f"q_{i}_options_display",
                 label_visibility="collapsed")
        st.markdown("---")

    st.subheader("🔑 Cevap Anahtarı")
    answers = exam_data.get("cevap_anahtari", {})
    cols = st.columns(len(answers) if len(answers) < 5 else 5)
    col_idx = 0
    for num, ans in answers.items():
         with cols[col_idx % len(cols)]:
              st.markdown(f"**Soru {num}:** {ans}")
         col_idx += 1
    st.markdown("--- --- --- --- ---") # Add a clear separator


def display_download_buttons(results_json_str, topic, source_pdf_name):
    """Displays download buttons for PDF and DOCX."""
    st.subheader(DOWNLOAD_HEADER)
    col1, col2 = st.columns(2)

    try:
        pdf_buffer = utils.create_pdf_exam(results_json_str, topic, source_pdf_name)
        with col1:
            st.download_button(
                label=DOWNLOAD_PDF_LABEL,
                data=pdf_buffer,
                file_name=f"{topic.replace(' ', '_')}_sinavi.pdf",
                mime="application/pdf"
            )
    except Exception as pdf_e:
        with col1:
            st.error(f"PDF oluşturulurken hata: {pdf_e}")

    try:
        docx_buffer = utils.create_docx_exam(results_json_str, topic, source_pdf_name)
        with col2:
            st.download_button(
                label=DOWNLOAD_DOCX_LABEL,
                data=docx_buffer,
                file_name=f"{topic.replace(' ', '_')}_sinavi.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    except Exception as docx_e:
        with col2:
            st.error(f"DOCX oluşturulurken hata: {docx_e}")

def display_exam_interface(results_json_str, topic, source_pdf_name):
    """Parses, displays, and provides download options for the exam."""
    exam_data = parse_exam_json(results_json_str)
    if exam_data:
        display_parsed_exam(exam_data)
        display_download_buttons(results_json_str, topic, source_pdf_name)
    # Error messages are handled within parse_exam_json and display_download_buttons

def process_uploaded_pdf(uploaded_file):
    """Saves and splits the uploaded PDF. Returns True on success, False otherwise."""
    try:
        with st.spinner("PDF kaydediliyor ve parçalara ayrılıyor..."):
            save_path = utils.save_uploaded_file(uploaded_file)
            if not save_path:
                 st.sidebar.error(f"'{uploaded_file.name}' kaydedilemedi.")
                 return False
            doc_chunks = utils.load_and_split_pdf(save_path)

        if doc_chunks:
            # Store chunks temporarily, generate a NEW unique ID for this processing attempt
            st.session_state[S_DOC_CHUNKS] = doc_chunks
            new_store_id = str(uuid.uuid4())
            st.session_state[S_CURRENT_STORE_ID] = new_store_id
            st.session_state[S_UPLOADED_FILENAME] = uploaded_file.name # Keep track of original filename
            # Mark vector store as NOT ready for this new ID yet
            st.session_state[S_VECTORSTORE_READY_FOR_ID] = None
            st.sidebar.success(f"'{uploaded_file.name}' başarıyla işlendi! ({len(doc_chunks)} parça).")
            return True
        else:
            st.sidebar.error("PDF işlenirken bir hata oluştu veya içerik bulunamadı.")
            # Reset states if processing fails
            st.session_state[S_DOC_CHUNKS] = None
            st.session_state[S_CURRENT_STORE_ID] = None
            return False
    except Exception as e:
        st.sidebar.error(f"PDF işleme hatası: {e}")
        st.session_state[S_DOC_CHUNKS] = None
        st.session_state[S_CURRENT_STORE_ID] = None
        return False

def create_vector_store_if_needed():
    """Creates the vector store for the currently assigned S_CURRENT_STORE_ID if it's not ready yet."""
    store_id = st.session_state.get(S_CURRENT_STORE_ID)
    ready_id = st.session_state.get(S_VECTORSTORE_READY_FOR_ID)
    chunks = st.session_state.get(S_DOC_CHUNKS)

    # Proceed only if we have a current ID, it's not already marked as ready, and we have chunks
    if store_id and store_id != ready_id and chunks:
        filename = st.session_state.get(S_UPLOADED_FILENAME, store_id) # Use filename for messages
        try:
            with st.spinner(f"'{filename}' için vektör deposu (ID: {store_id[:8]}...) oluşturuluyor/güncelleniyor..."):
                # force_recreate=True ensures deletion of old dir for this specific UUID store_id
                rag_pipeline.create_vectorstore(chunks, store_id=store_id, force_recreate=True)
            # If successful, mark this store ID as ready
            st.session_state[S_VECTORSTORE_READY_FOR_ID] = store_id
            # Clear chunks as they are now in the vector store
            st.session_state[S_DOC_CHUNKS] = None
        except (ValueError, RuntimeError) as ve:
            st.sidebar.error(f"Vektör deposu oluşturma hatası ({filename}, ID: {store_id[:8]}...): {ve}")
            # Keep S_VECTORSTORE_READY_FOR_ID as None or its previous value
        except Exception as e:
            st.sidebar.error(f"Beklenmedik vektör deposu oluşturma hatası ({filename}, ID: {store_id[:8]}...): {e}")

def handle_pdf_upload():
    """Manages the PDF upload section in the sidebar."""
    uploaded_pdf = st.sidebar.file_uploader(PDF_UPLOAD_LABEL, type=["pdf"], key="pdf_uploader")

    if uploaded_pdf:
        # Check if it's a different file than the last one successfully processed
        # or if the vector store for the current ID isn't ready
        current_store_id = st.session_state.get(S_CURRENT_STORE_ID)
        ready_store_id = st.session_state.get(S_VECTORSTORE_READY_FOR_ID)
        is_new_file = st.session_state.get(S_UPLOADED_FILENAME) != uploaded_pdf.name
        needs_processing = is_new_file or (current_store_id != ready_store_id)

        if needs_processing:
            st.sidebar.info(f"Dosya '{uploaded_pdf.name}' işleniyor...")
            # Resetting here is crucial if it's a truly new file
            # If it's a retry for the same file, process_uploaded_pdf will assign new ID
            # reset_exam_state() # Maybe reset only specific parts?
            st.session_state[S_EXAM_RESULTS] = None # Clear previous exam results
            process_uploaded_pdf(uploaded_file=uploaded_pdf)
            # Vector store creation will be attempted by display_sidebar calls
        else:
            # Same file, and vector store is marked as ready for its ID
            filename = st.session_state.get(S_UPLOADED_FILENAME)
            store_id_short = current_store_id[:8] if current_store_id else "N/A"
            st.sidebar.success(f"'{filename}' zaten yüklü ve vektör deposu (ID: {store_id_short}...) hazır.")

def display_sidebar():
    """Displays all elements in the Streamlit sidebar."""
    st.sidebar.header(SIDEBAR_HEADER)

    # 1. PDF Upload Section
    handle_pdf_upload()

    # Attempt to create vector store for the current ID if needed
    create_vector_store_if_needed()

    # 2. Exam Parameters Section
    st.sidebar.subheader(EXAM_PARAMS_HEADER)
    exam_topic = st.sidebar.text_input(TOPIC_INPUT_LABEL,
                                       placeholder=TOPIC_INPUT_PLACEHOLDER,
                                       key="topic_input")
    exam_level = st.sidebar.selectbox(LEVEL_SELECT_LABEL, LEVEL_OPTIONS, key="level_input")
    num_questions = st.sidebar.number_input(NUM_QUESTIONS_LABEL,
                                            min_value=1,
                                            max_value=MAX_QUESTIONS,
                                            value=DEFAULT_QUESTIONS,
                                            key="num_q_input")

    # Determine if exam generation is possible
    current_store_id = st.session_state.get(S_CURRENT_STORE_ID)
    ready_store_id = st.session_state.get(S_VECTORSTORE_READY_FOR_ID)
    pdf_ready = current_store_id is not None and current_store_id == ready_store_id
    topic_provided = bool(exam_topic)
    can_generate = pdf_ready or topic_provided

    # 3. Generate Exam Button
    if st.sidebar.button(GENERATE_BUTTON_LABEL, disabled=not can_generate, key="generate_button"):
        handle_exam_generation(pdf_ready, topic_provided, exam_topic, exam_level, num_questions)

    # 4. History Section
    display_history()

def handle_exam_generation(pdf_is_ready, topic_provided, exam_topic, exam_level, num_questions):
    """Handles the logic when the 'Generate Exam' button is clicked."""

    store_id = st.session_state.get(S_CURRENT_STORE_ID) # Get the ID of the currently loaded PDF's store
    filename = st.session_state.get(S_UPLOADED_FILENAME) # Get the original filename for display

    # Determine generation mode
    if pdf_is_ready and store_id:
        actual_topic = exam_topic if exam_topic else DEFAULT_TOPIC_NAME
        generation_mode = "PDF İçeriğinden"
        source_pdf_display_name = filename # Use original filename for display/saving
    elif topic_provided:
        actual_topic = exam_topic
        generation_mode = "Konu Bazlı"
        source_pdf_display_name = None
        store_id = None # No specific store ID for topic-only mode
    else:
        st.error("Sınav oluşturmak için ya işlenmiş bir PDF seçin ya da bir konu girin.")
        return

    st.session_state[S_EXAM_RESULTS] = None
    st.session_state[S_EXAM_TO_DISPLAY] = None

    st.info(f"'{actual_topic}' konusunda ({generation_mode}) {num_questions} adet {exam_level.lower()} seviye sınav oluşturuluyor...")

    try:
        llm = rag_pipeline.get_llm()
        response_str = ""

        with st.spinner("Sorular Gemini API ile üretiliyor... Lütfen bekleyin."):
            if pdf_is_ready and store_id:
                # Get retriever using the specific store_id for the current PDF
                retriever = rag_pipeline.get_retriever(store_id=store_id)
                rag_chain = rag_pipeline.create_rag_chain(retriever, llm)
                response_str = rag_chain.invoke({
                    "topic": actual_topic,
                    "num_questions": num_questions,
                    "level": exam_level
                })
            elif topic_provided:
                topic_chain = rag_pipeline.create_topic_only_chain(llm)
                response_str = topic_chain.invoke({
                    "topic": actual_topic,
                    "num_questions": num_questions,
                    "level": exam_level
                })

        st.session_state[S_EXAM_RESULTS] = response_str
        st.session_state[S_LAST_GENERATED_TOPIC] = actual_topic
        # Save original filename in session state for display/db
        st.session_state[S_LAST_SOURCE_PDF] = source_pdf_display_name

        save_exam_results(response_str, actual_topic, num_questions, source_pdf_display_name)

    except ValueError as ve:
         st.error(f"Sınav oluşturma hatası: {ve}. API Anahtarınızı kontrol edin.")
    except RuntimeError as re:
         # Include store_id in the error message if relevant
         store_id_msg = f" (ID: {store_id[:8]}...)" if store_id else ""
         st.error(f"Sınav oluşturma hatası: {re}. Vektör deposu{store_id_msg} hazır mı/yüklenebildi mi?")
    except Exception as e:
        st.error(f"Sınav oluşturulurken beklenmedik bir hata oluştu: {e}", exc_info=True) # Show traceback for unexpected errors
        st.error("Lütfen API anahtarınızın doğru olduğundan ve varsa PDF dosyasının geçerli olduğundan emin olun.")

def save_exam_results(response_str, topic, num_questions, pdf_name):
    """Parses the LLM response and saves the exam to the database."""
    questions_json = json.dumps([{"raw_response": response_str}]) # Default if parsing fails
    answers_json = json.dumps({})
    parse_error = False

    try:
        # Check if the response looks like JSON before attempting to parse
        if response_str and (response_str.strip().startswith("{") or response_str.strip().startswith("```json")):
            parsed_results = parse_exam_json(response_str)
            if parsed_results:
                # Topic-only chain might not include 'kaynak', handle missing keys gracefully
                questions_json = json.dumps(parsed_results.get("sorular", []), ensure_ascii=False)
                answers_json = json.dumps(parsed_results.get("cevap_anahtari", {}), ensure_ascii=False)
            else:
                 # parse_exam_json already showed an error, mark as parse error
                 parse_error = True
        else:
            # Response is not JSON, treat as raw text
            st.warning("Oluşturulan yanıt JSON formatında değil. Ham metin olarak kaydedilecek.")
            parse_error = True # Consider non-JSON also a form of parse error for saving

        # Save exam regardless of parsing success, storing raw response if needed
        utils.save_exam(
            session_id=st.session_state.get(S_SESSION_ID),
            pdf_name=pdf_name,
            topic=topic,
            num_questions=num_questions,
            questions_json=questions_json,
            answers_json=answers_json
        )
        if not parse_error:
            st.success("Sınav başarıyla oluşturuldu ve veritabanına kaydedildi.")
        # If parse_error is True, user already saw a warning/error from parsing functions

    except Exception as db_err:
        st.error(f"Sınav veritabanına kaydedilemedi: {db_err}")

def display_results_area():
    """Displays either the selected historical exam or the newly generated one."""
    # Check if we need to display a specific historical exam
    if st.session_state.get(S_EXAM_TO_DISPLAY):
        exam_detail = st.session_state[S_EXAM_TO_DISPLAY]

        reconstructed_exam_str = None
        try:
            # Ensure defaults are used if JSON fields are null/missing in DB
            questions_part = json.loads(exam_detail.get('questions_json') or '[]')
            answers_part = json.loads(exam_detail.get('answers_json') or '{}')
            full_exam_dict = {
                "sorular": questions_part,
                "cevap_anahtari": answers_part
            }
            # Check if the questions part contains the raw response indicator
            is_raw = any(q.get("raw_response") for q in questions_part if isinstance(q, dict))
            if is_raw:
                 # Handle case where questions_json stored raw response
                 raw_response_text = questions_part[0].get("raw_response", "Ham veri bulunamadı.") if questions_part else "Ham veri bulunamadı."
                 st.warning("Bu sınavın orijinal çıktısı JSON formatında değildi. Ham metin gösteriliyor:")
                 st.code(raw_response_text, language='text')
                 # Prevent trying to display JSON
                 reconstructed_exam_str = None
            else:
                 reconstructed_exam_str = json.dumps(full_exam_dict, ensure_ascii=False, indent=2)

        except json.JSONDecodeError as json_err:
            st.error(f"Geçmiş sınav verisi yüklenirken JSON ayrıştırma hatası: {json_err}")
            st.text("Sorular Ham Veri:")
            st.code(exam_detail.get('questions_json', 'Yok'), language='json')
            st.text("Cevaplar Ham Veri:")
            st.code(exam_detail.get('answers_json', 'Yok'), language='json')
            reconstructed_exam_str = None # Ensure we don't try to display if error
        except Exception as e: # Catch other potential errors during processing
             st.error(f"Geçmiş sınav verisi işlenirken hata: {e}")
             st.code(exam_detail, language='python') # Show the raw detail dictionary
             reconstructed_exam_str = None


        # Extract necessary info for display_exam_interface
        topic = exam_detail.get('topic', 'Bilinmeyen Konu')
        # Handle None pdf_name from DB for topic-only exams
        pdf_name = exam_detail.get('pdf_name') or TOPIC_ONLY_DISPLAY_NAME

        # Pass the reconstructed JSON string to the display function if valid
        if reconstructed_exam_str:
            display_exam_interface(reconstructed_exam_str, topic, pdf_name)
        # else: Error/warning already shown inside the try block or for raw data

    # Display newly generated exam results if available and no historical exam selected
    elif st.session_state.get(S_EXAM_RESULTS):
        display_topic = st.session_state.get(S_LAST_GENERATED_TOPIC, 'Sinav')
        # Use the source PDF stored during generation
        display_pdf_name = st.session_state.get(S_LAST_SOURCE_PDF) or TOPIC_ONLY_DISPLAY_NAME
        display_exam_interface(st.session_state[S_EXAM_RESULTS], display_topic, display_pdf_name)


def display_history():
    """Displays the exam history section in the sidebar."""
    st.sidebar.subheader(HISTORY_HEADER)
    try:
        history = utils.get_exam_history(limit=HISTORY_LIMIT)
        if not history:
            st.sidebar.caption("Geçmiş sınav kaydı bulunamadı.")
            return

        for entry in history:
            exam_id = entry['id']
            timestamp_str = entry.get('timestamp', 'Bilinmeyen Tarih')
            try:
                # Parse the ISO string, assuming it's UTC (often indicated by Z or +00:00 implicitly in DB)
                timestamp_utc = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                # Ensure it's timezone-aware (set to UTC if naive)
                if timestamp_utc.tzinfo is None:
                    timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
                # Convert to Turkey Time (UTC+3)
                timestamp_tr = timestamp_utc.astimezone(TURKEY_TZ)
                # Format the TR time
                display_time = timestamp_tr.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                 # Fallback for potentially non-ISO formats
                 # Try basic parsing, assuming it might be local time already (less reliable)
                 try:
                     naive_dt = datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                     display_time = naive_dt.strftime("%Y-%m-%d %H:%M") + " (Format Hatalı?)"
                 except ValueError:
                      display_time = timestamp_str.split('.')[0].replace('T', ' ') + " (Format Bilinmiyor)"

            topic = entry.get('topic', 'Bilinmeyen Konu')
            num_q = entry.get('num_questions', '?')
            pdf_display = entry.get('pdf_name')

            expander_title = f"{display_time} - {topic} ({num_q} soru)"
            with st.sidebar.expander(expander_title):
                if pdf_display:
                    st.caption(f"Kaynak: {pdf_display}")
                else:
                    st.caption(f"Kaynak: {TOPIC_ONLY_DISPLAY_NAME}")

                col1_actions, col2_actions = st.columns(2)
                with col1_actions:
                    if st.button(VIEW_BUTTON_LABEL, key=f"view_{exam_id}", use_container_width=True):
                        full_exam_data = utils.get_exam_by_id(exam_id)
                        if full_exam_data:
                            st.session_state[S_EXAM_TO_DISPLAY] = full_exam_data
                            st.session_state[S_EXAM_RESULTS] = None
                            st.rerun()
                        else:
                            st.error(f"Sınav ID {exam_id} bulunamadı.")

                with col2_actions:
                    if st.button(DELETE_BUTTON_LABEL, key=f"delete_{exam_id}", use_container_width=True):
                        deleted = utils.delete_exam(exam_id)
                        if deleted:
                            st.success(f"Sınav ID {exam_id} silindi.")
                            if st.session_state.get(S_EXAM_TO_DISPLAY) and st.session_state[S_EXAM_TO_DISPLAY].get('id') == exam_id:
                                st.session_state[S_EXAM_TO_DISPLAY] = None
                            st.rerun()
                        else:
                            st.error(f"Sınav ID {exam_id} silinirken hata.")

    except Exception as e:
        st.sidebar.error(f"Geçmiş sınavlar yüklenemedi: {e}")


# --- Main Application Flow ---

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(APP_TITLE)
    st.write(APP_DESC)

    # Initialization
    _ = load_api_key()
    initialize_database()
    initialize_session_state()

    # Sidebar sections
    display_sidebar()

    # Main content area for displaying exams
    display_results_area()

if __name__ == "__main__":
    main() 