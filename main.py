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
from datetime import datetime
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
    """Saves, splits the uploaded PDF and updates session state."""
    try:
        with st.spinner("PDF kaydediliyor ve parçalara ayrılıyor..."):
            save_path = utils.save_uploaded_file(uploaded_file)
            doc_chunks = utils.load_and_split_pdf(save_path)

        if doc_chunks:
            st.session_state[S_DOC_CHUNKS] = doc_chunks
            st.session_state[S_PDF_PROCESSED] = True
            st.session_state[S_VECTORSTORE_CREATED] = False # Needs recreation
            st.sidebar.success(f"'{uploaded_file.name}' başarıyla işlendi! ({len(doc_chunks)} parça)")
            return True
        else:
            st.sidebar.error("PDF işlenirken bir hata oluştu veya içerik bulunamadı.")
            st.session_state[S_PDF_PROCESSED] = False
            return False
    except Exception as e:
        st.sidebar.error(f"PDF işleme hatası: {e}")
        st.session_state[S_PDF_PROCESSED] = False
        return False

def create_vector_store_if_needed():
    """Creates the vector store if PDF is processed and store doesn't exist."""
    if st.session_state[S_PDF_PROCESSED] and not st.session_state[S_VECTORSTORE_CREATED]:
        try:
            with st.spinner("Embeddingler oluşturuluyor ve vektör deposu hazırlanıyor..."):
                # Pass force_recreate=True because we processed a new file (or re-processed)
                rag_pipeline.create_vectorstore(st.session_state[S_DOC_CHUNKS], force_recreate=True)
            st.session_state[S_VECTORSTORE_CREATED] = True
        except ValueError as ve:
            st.sidebar.error(f"Vektör deposu oluşturma hatası: {ve}. API Anahtarınızı kontrol edin.")
            st.session_state[S_VECTORSTORE_CREATED] = False
        except Exception as e:
            st.sidebar.error(f"Vektör deposu oluşturma hatası: {e}")
            st.session_state[S_VECTORSTORE_CREATED] = False

def handle_pdf_upload():
    """Manages the PDF upload section in the sidebar."""
    uploaded_pdf = st.sidebar.file_uploader(PDF_UPLOAD_LABEL, type=["pdf"], key="pdf_uploader")

    if uploaded_pdf:
        # Check if it's a new file or reprocessing is needed
        new_file_uploaded = st.session_state.get(S_UPLOADED_FILENAME) != uploaded_pdf.name
        needs_reprocessing = not st.session_state.get(S_PDF_PROCESSED)

        if new_file_uploaded:
            st.sidebar.info(f"Yeni dosya algılandı: '{uploaded_pdf.name}'. İşleniyor...")
            # Reset relevant states for the new file
            reset_exam_state()
            st.session_state[S_UPLOADED_FILENAME] = uploaded_pdf.name
            process_uploaded_pdf(uploaded_pdf)
            # Vector store creation will be triggered automatically after this if successful

        elif needs_reprocessing:
            st.sidebar.warning(f"'{uploaded_pdf.name}' daha önce yüklenmişti ancak işlenmemiş görünüyor. Tekrar işleniyor...")
            # Ensure filename is set, process, and trigger vector store creation
            st.session_state[S_UPLOADED_FILENAME] = uploaded_pdf.name
            process_uploaded_pdf(uploaded_pdf)

        else:
            # File is the same and already processed
            st.sidebar.success(f"'{uploaded_pdf.name}' zaten yüklü ve işlenmiş.")
            # Ensure vector store is created if it was somehow lost
            create_vector_store_if_needed()

def display_sidebar():
    """Displays all elements in the Streamlit sidebar."""
    st.sidebar.header(SIDEBAR_HEADER)

    # 1. PDF Upload Section
    handle_pdf_upload()

    # Automatically trigger vector store creation after PDF processing step if needed
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
    pdf_ready = st.session_state.get(S_PDF_PROCESSED, False) and st.session_state.get(S_VECTORSTORE_CREATED, False)
    topic_provided = bool(exam_topic)
    can_generate = pdf_ready or topic_provided

    # 3. Generate Exam Button
    if st.sidebar.button(GENERATE_BUTTON_LABEL, disabled=not can_generate, key="generate_button"):
        handle_exam_generation(pdf_ready, topic_provided, exam_topic, exam_level, num_questions)

    # 4. History Section
    display_history()


def handle_exam_generation(pdf_available, topic_only_mode_possible, exam_topic, exam_level, num_questions):
    """Handles the logic when the 'Generate Exam' button is clicked."""
    # Re-evaluate conditions strictly inside the handler
    pdf_ready = st.session_state.get(S_PDF_PROCESSED, False) and st.session_state.get(S_VECTORSTORE_CREATED, False)
    topic_provided = bool(exam_topic)

    # Determine generation mode
    if pdf_ready:
        actual_topic = exam_topic if exam_topic else DEFAULT_TOPIC_NAME
        generation_mode = "PDF İçeriğinden"
        source_pdf_display_name = st.session_state.get(S_UPLOADED_FILENAME)
    elif topic_provided:
        actual_topic = exam_topic
        generation_mode = "Konu Bazlı"
        source_pdf_display_name = None
    else:
        st.error("Sınav oluşturmak için ya PDF yükleyin ya da bir konu girin.") # Should not happen due to button disable logic
        return

    # Clear previous results and flags
    st.session_state[S_EXAM_RESULTS] = None
    st.session_state[S_EXAM_TO_DISPLAY] = None

    st.info(f"'{actual_topic}' konusunda ({generation_mode}) {num_questions} adet {exam_level.lower()} seviye sınav oluşturuluyor...")

    try:
        llm = rag_pipeline.get_llm()
        response_str = ""

        with st.spinner("Sorular Gemini API ile üretiliyor... Lütfen bekleyin."):
            if pdf_ready:
                retriever = rag_pipeline.get_retriever()
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
        st.session_state[S_LAST_SOURCE_PDF] = source_pdf_display_name

        # Attempt to parse and save immediately after generation
        save_exam_results(response_str, actual_topic, num_questions, source_pdf_display_name)

    except ValueError as ve:
         st.error(f"Sınav oluşturma hatası: {ve}. API Anahtarınızı kontrol edin.")
    except RuntimeError as re:
         st.error(f"Sınav oluşturma hatası: {re}. Vektör deposu hazır mı?")
    except Exception as e:
        st.error(f"Sınav oluşturulurken beklenmedik bir hata oluştu: {e}")
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
                # Handle potential timezone info if present (e.g., +00:00)
                timestamp_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                # Format to local time might be better, but requires timezone awareness
                # For simplicity, stick to UTC or the stored format
                display_time = timestamp_dt.strftime("%Y-%m-%d %H:%M") # Consider adding timezone info if relevant
            except ValueError:
                 # Fallback for potentially non-ISO formats
                display_time = timestamp_str.split('.')[0].replace('T', ' ')

            topic = entry.get('topic', 'Bilinmeyen Konu')
            num_q = entry.get('num_questions', '?')
            pdf_display = entry.get('pdf_name')

            # Use columns for better layout within the expander
            expander_title = f"{display_time} - {topic} ({num_q} soru)"
            with st.sidebar.expander(expander_title):
                # Display source info
                if pdf_display:
                    st.caption(f"Kaynak: {pdf_display}")
                else:
                    st.caption(f"Kaynak: {TOPIC_ONLY_DISPLAY_NAME}")

                # Action buttons
                col1_actions, col2_actions = st.columns(2)
                with col1_actions:
                    if st.button(VIEW_BUTTON_LABEL, key=f"view_{exam_id}", use_container_width=True):
                        full_exam_data = utils.get_exam_by_id(exam_id)
                        if full_exam_data:
                            # Store the full data to be displayed after rerun
                            st.session_state[S_EXAM_TO_DISPLAY] = full_exam_data
                            # Clear any newly generated results to avoid conflict
                            st.session_state[S_EXAM_RESULTS] = None
                            st.rerun()
                        else:
                            st.error(f"Sınav ID {exam_id} bulunamadı.")

                with col2_actions:
                    if st.button(DELETE_BUTTON_LABEL, key=f"delete_{exam_id}", use_container_width=True):
                        deleted = utils.delete_exam(exam_id)
                        if deleted:
                            st.success(f"Sınav ID {exam_id} silindi.")
                            # Clear display if the deleted exam was showing
                            if st.session_state.get(S_EXAM_TO_DISPLAY) and st.session_state[S_EXAM_TO_DISPLAY].get('id') == exam_id:
                                st.session_state[S_EXAM_TO_DISPLAY] = None
                            # Clear current results if they happened to be from the deleted one (less likely but possible)
                            # This requires storing the ID of the last generated exam if we want perfect cleanup
                            # For now, just rerun to refresh the history view
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