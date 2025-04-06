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
from dotenv import load_dotenv
import utils
import rag_pipeline
from datetime import datetime
import uuid # For session ID generation

# Load environment variables (especially Google API Key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Try streamlit secrets if .env fails (for deployment)
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, AttributeError):
        st.error("Google API Anahtarı bulunamadı. Lütfen .env dosyasına veya Streamlit secrets'a ekleyin.")
        st.stop()

# Initialize Database
try:
    utils.init_db()
    # st.sidebar.success("Veritabanı bağlantısı başarılı.") # Can be removed for cleaner UI
except Exception as e:
    st.sidebar.error(f"Veritabanı başlatılamadı: {e}")
    st.stop()

st.set_page_config(page_title="Sınav Hazırlayan Chatbot", page_icon="🚀")

st.title("🚀 Sınav Hazırlayan RAG Based Chatbot")

st.write("""
Kendi ders notlarınızdan veya dokümanlarınızdan hızlıca çoktan seçmeli sınavlar oluşturun!

**Nasıl Kullanılır:**
1.  Sol menüden PDF dosyanızı yükleyin.
2.  Yine sol menüden sınavınız için istediğiniz ayarları (konu, soru sayısı vb.) belirtin.
3.  'Sınav Oluştur' butonuna tıklayın.
4.  Chatbot sizin için soruları hazırlayacaktır. (Cevap anahtarı ile birlikte!)
""")

# --- Session State Initialization ---
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'doc_chunks' not in st.session_state:
    st.session_state.doc_chunks = None
if 'vectorstore_created' not in st.session_state:
    st.session_state.vectorstore_created = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Generate a unique ID for this session
if 'exam_results' not in st.session_state:
    st.session_state.exam_results = None
if 'exam_to_display' not in st.session_state:
    st.session_state.exam_to_display = None # Stores full data of exam to show

# --- Helper function to display exam results ---
def display_exam(results_json_str, topic, source_pdf_name):
    try:
        # Try cleaning potential markdown code fences first
        cleaned_response = results_json_str.strip().removeprefix("```json").removesuffix("```").strip()
        results = json.loads(cleaned_response)
        if not isinstance(results, dict) or "sorular" not in results or "cevap_anahtari" not in results:
             st.error("Oluşturulan sınav formatı anlaşılamadı. Ham veri aşağıdadır:")
             st.code(results_json_str, language='json')
             return # Stop processing if format is wrong

        st.subheader("📝 Oluşturulan Sınav")
        for i, q in enumerate(results.get("sorular", [])):
            st.markdown(f"**Soru {q.get('soru_no', i+1)}:** {q.get('soru_metni', 'Soru metni bulunamadı')}")
            kaynak = q.get('kaynak', None)
            if kaynak:
                st.caption(f"Kaynak: {kaynak}")
            options = q.get("secenekler", {})
            st.radio("Seçenekler:",
                     options=[f"{key}: {value}" for key, value in options.items()],
                     key=f"q_{i}_options", label_visibility="collapsed")
            st.markdown("---")

        st.subheader("🔑 Cevap Anahtarı")
        answers = results.get("cevap_anahtari", {})
        cols = st.columns(len(answers) if len(answers) < 5 else 5)
        col_idx = 0
        for num, ans in answers.items():
             with cols[col_idx % len(cols)]:
                  st.markdown(f"**Soru {num}:** {ans}")
             col_idx += 1

        st.markdown("--- --- --- --- ---") # Add a clear separator

        # --- Download Buttons ---
        st.subheader("💾 Sınavı İndir")
        col1, col2 = st.columns(2)

        try:
            # Create PDF in memory
            pdf_buffer = utils.create_pdf_exam(results_json_str, topic, source_pdf_name)
            with col1:
                st.download_button(
                    label="📄 PDF olarak İndir",
                    data=pdf_buffer,
                    file_name=f"{topic.replace(' ', '_')}_sinavi.pdf",
                    mime="application/pdf"
                )
        except Exception as pdf_e:
            with col1:
                st.error(f"PDF oluşturulurken hata: {pdf_e}")

        try:
            # Create DOCX in memory
            docx_buffer = utils.create_docx_exam(results_json_str, topic, source_pdf_name)
            with col2:
                st.download_button(
                    label="📝 Word (DOCX) olarak İndir",
                    data=docx_buffer,
                    file_name=f"{topic.replace(' ', '_')}_sinavi.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        except Exception as docx_e:
            with col2:
                st.error(f"DOCX oluşturulurken hata: {docx_e}")

    except json.JSONDecodeError:
        st.error("Üretilen yanıt JSON formatında değil, dışa aktarılamıyor. Ham yanıt aşağıdadır:")
        st.text(results_json_str)
    except Exception as e:
        st.error(f"Sınav gösterilirken/dışa aktarılırken bir hata oluştu: {e}")
        st.text("Ham veri:")
        st.code(results_json_str, language='text')


# --- Sidebar --- #
st.sidebar.header("Ayarlar")

# PDF Yükleme
uploaded_pdf = st.sidebar.file_uploader("1. PDF Yükle", type=["pdf"], key="pdf_uploader")

if uploaded_pdf:
    # Check if it's a new file
    if st.session_state.uploaded_filename != uploaded_pdf.name:
        st.session_state.pdf_processed = False
        st.session_state.vectorstore_created = False
        st.session_state.exam_results = None # Clear previous results
        st.session_state.uploaded_filename = uploaded_pdf.name
        
        try:
            with st.spinner("PDF kaydediliyor ve parçalara ayrılıyor..."):
                # Save using the utility function
                save_path = utils.save_uploaded_file(uploaded_pdf)
                st.session_state.doc_chunks = utils.load_and_split_pdf(save_path)
                # No need to remove the saved file, keep it for potential re-use or reference

            if st.session_state.doc_chunks:
                st.sidebar.success(f"'{uploaded_pdf.name}' başarıyla işlendi! ({len(st.session_state.doc_chunks)} parça)")
                st.session_state.pdf_processed = True
            else:
                st.sidebar.error("PDF işlenirken bir hata oluştu veya içerik bulunamadı.")
                st.session_state.pdf_processed = False

        except Exception as e:
            st.sidebar.error(f"PDF işleme hatası: {e}")
            st.session_state.pdf_processed = False
    else:
         # File with the same name re-uploaded or already present
         # Ensure flags are correct, especially if the app restarted
         if not st.session_state.pdf_processed:
             st.sidebar.warning(f"'{uploaded_pdf.name}' daha önce yüklenmişti ancak işlenmemiş görünüyor. Tekrar işleniyor...")
             # Repeat processing logic if needed (or assume state loss and re-process)
             try:
                 with st.spinner("PDF yeniden işleniyor..."):
                    save_path = utils.save_uploaded_file(uploaded_pdf) # Re-save or find existing
                    st.session_state.doc_chunks = utils.load_and_split_pdf(save_path)
                 if st.session_state.doc_chunks:
                     st.sidebar.success(f"'{uploaded_pdf.name}' başarıyla yeniden işlendi! ({len(st.session_state.doc_chunks)} parça)")
                     st.session_state.pdf_processed = True
                 else:
                     st.sidebar.error("PDF yeniden işlenirken bir hata oluştu.")
                     st.session_state.pdf_processed = False
             except Exception as e:
                st.sidebar.error(f"PDF yeniden işleme hatası: {e}")
                st.session_state.pdf_processed = False
         else:
             st.sidebar.success(f"'{uploaded_pdf.name}' zaten yüklü ve işlenmiş.")

# --- Vector Store Creation (Triggered after PDF processing) ---
# Only create vectorstore if PDF is processed AND it hasn't been created yet
if st.session_state.pdf_processed and not st.session_state.vectorstore_created:
     
     try:
         with st.spinner("Embeddingler oluşturuluyor ve vektör deposu hazırlanıyor..."):
             # Pass force_recreate=True because we processed a new file (or re-processed)
             rag_pipeline.create_vectorstore(st.session_state.doc_chunks, force_recreate=True)
         
         st.session_state.vectorstore_created = True
     except ValueError as ve:
         st.sidebar.error(f"Vektör deposu oluşturma hatası: {ve}. API Anahtarınızı kontrol edin.")
         st.session_state.vectorstore_created = False
     except Exception as e:
         st.sidebar.error(f"Vektör deposu oluşturma hatası: {e}")
         st.session_state.vectorstore_created = False

# Sınav Parametreleri
st.sidebar.subheader("2. Sınav Özellikleri")
exam_topic = st.sidebar.text_input("Konu (PDF yüklemeden de sadece konuyla sınav oluşturabilirsiniz)", placeholder="Örn: Türkiye Coğrafyası", key="topic_input")
exam_level = st.sidebar.selectbox("Seviye", ["Kolay", "Orta", "Zor"], key="level_input")
num_questions = st.sidebar.number_input("Soru Sayısı", min_value=1, max_value=20, value=5, key="num_q_input")

# Determine if exam generation is possible
# Need either (processed PDF) OR (a topic entered)
can_generate = (st.session_state.pdf_processed and st.session_state.vectorstore_created) or bool(exam_topic)

if st.sidebar.button("3. Sınav Oluştur", disabled=not can_generate, key="generate_button"):

    # Double check conditions met inside the button logic
    pdf_available = st.session_state.pdf_processed and st.session_state.vectorstore_created
    topic_only_mode = not pdf_available and bool(exam_topic)

    if pdf_available or topic_only_mode:
        st.session_state.exam_results = None # Clear previous results
        st.session_state.exam_to_display = None # Clear historical display flag

        # Determine the actual topic to use (either user input or generic from PDF)
        if topic_only_mode:
            actual_topic = exam_topic # Must have a topic in this mode
            generation_mode = "Konu Bazlı"
            source_pdf_display_name = None # No source PDF
        elif pdf_available:
            actual_topic = exam_topic if exam_topic else "Belgenin Geneli"
            generation_mode = "PDF İçeriğinden"
            source_pdf_display_name = st.session_state.uploaded_filename # Use uploaded PDF name
        else:
            st.error("Sınav oluşturmak için ya PDF yükleyin ya da bir konu girin.") # Should not happen due to button disable logic
            st.stop()

        st.info(f"'{actual_topic}' konusunda ({generation_mode}) {num_questions} adet {exam_level.lower()} seviye sınav oluşturuluyor...")

        try:
            llm = rag_pipeline.get_llm()
            response_str = ""

            with st.spinner("Sorular Gemini API ile üretiliyor... Lütfen bekleyin."):
                if pdf_available:
                    # Use RAG chain
                    retriever = rag_pipeline.get_retriever()
                    rag_chain = rag_pipeline.create_rag_chain(retriever, llm)
                    response_str = rag_chain.invoke({
                        "topic": actual_topic,
                        "num_questions": num_questions,
                        "level": exam_level
                    })
                elif topic_only_mode:
                    # Use Topic-Only chain
                    topic_chain = rag_pipeline.create_topic_only_chain(llm)
                    response_str = topic_chain.invoke({
                        "topic": actual_topic,
                        "num_questions": num_questions,
                        "level": exam_level
                    })

            st.session_state.exam_results = response_str
            st.session_state.last_generated_topic = actual_topic # Store topic for display/download
            # Store PDF name (or None) used for generation for display/download
            st.session_state.last_source_pdf = source_pdf_display_name

            # Attempt to parse and save immediately after generation
            try:
                # ... (JSON parsing logic - slight adjustment for topic-only might be needed if format differs)
                if response_str.strip().startswith("{") or response_str.strip().startswith("```json"):
                    # ... (cleaning and parsing) ...
                    cleaned_response = response_str.strip().removeprefix("```json").removesuffix("```").strip()
                    parsed_results = json.loads(cleaned_response)
                    # Topic-only chain doesn't include 'kaynak', default to empty list/dict if missing
                    questions_json = json.dumps(parsed_results.get("sorular", []))
                    answers_json = json.dumps(parsed_results.get("cevap_anahtari", {}))
                else:
                    # ... (handling non-JSON response) ...
                    questions_json = json.dumps([{"raw_response": response_str}])
                    answers_json = json.dumps({})

                # Save with the actual topic used and potentially None for pdf_name
                utils.save_exam(
                    session_id=st.session_state.session_id,
                    pdf_name=source_pdf_display_name, # Will be None for topic-only
                    topic=actual_topic,
                    num_questions=num_questions,
                    questions_json=questions_json,
                    answers_json=answers_json
                )
                st.success("Sınav başarıyla oluşturuldu ve veritabanına kaydedildi.")
            except json.JSONDecodeError as json_e:
                st.error(f"Oluşturulan sınavın formatı (JSON) ayrıştırılamadı: {json_e}. Sınav veritabanına ham metin olarak kaydedilecek.")
                utils.save_exam(
                    session_id=st.session_state.session_id,
                    pdf_name=source_pdf_display_name,
                    topic=actual_topic,
                    num_questions=num_questions,
                    questions_json=json.dumps([{"raw_response": response_str}]),
                    answers_json=json.dumps({})
                 )
            except Exception as db_err:
                st.error(f"Sınav veritabanına kaydedilemedi: {db_err}")

        except ValueError as ve:
             st.error(f"Sınav oluşturma hatası: {ve}. API Anahtarınızı kontrol edin.")
        except RuntimeError as re:
             st.error(f"Sınav oluşturma hatası: {re}. Vektör deposu hazır mı?")
        except Exception as e:
            st.error(f"Sınav oluşturulurken beklenmedik bir hata oluştu: {e}")
            st.error("Lütfen API anahtarınızın doğru olduğundan ve PDF dosyasının geçerli olduğundan emin olun.")

    else:
        # Feedback if button was clicked while disabled (more specific now)
        if not st.session_state.pdf_processed and not bool(exam_topic):
            st.sidebar.warning("Lütfen bir PDF yükleyin VEYA bir konu girin.")
        elif st.session_state.pdf_processed and not st.session_state.vectorstore_created:
            st.sidebar.warning("PDF işlendi ancak vektör deposu henüz hazır değil, lütfen bekleyin.")
        # Other potential edge cases?

# --- Display Exam Results Area ---
# Check if we need to display a specific historical exam
if st.session_state.exam_to_display:
    exam_detail = st.session_state.exam_to_display
    st.session_state.exam_to_display = None # Clear after loading

    # Reconstruct the full JSON structure expected by display_exam
    reconstructed_exam_str = None
    try:
        questions_part = json.loads(exam_detail.get('questions_json', '[]')) # Default to empty list
        answers_part = json.loads(exam_detail.get('answers_json', '{}'))   # Default to empty dict
        full_exam_dict = {
            "sorular": questions_part,
            "cevap_anahtari": answers_part
        }
        reconstructed_exam_str = json.dumps(full_exam_dict, ensure_ascii=False, indent=2) # Convert back to string
    except json.JSONDecodeError as json_err:
        st.error(f"Geçmiş sınav verisi JSON formatında değil: {json_err}")
        # Optionally show raw data
        st.text("Sorular Ham Veri:")
        st.code(exam_detail.get('questions_json', 'Yok'), language='json')
        st.text("Cevaplar Ham Veri:")
        st.code(exam_detail.get('answers_json', 'Yok'), language='json')

    # Extract necessary info for display_exam
    topic = exam_detail.get('topic', 'Bilinmeyen Konu')
    pdf_name = exam_detail.get('pdf_name', 'Konu Bazlı') # Handle None pdf_name

    # Pass the reconstructed JSON string to display_exam
    if reconstructed_exam_str:
        display_exam(reconstructed_exam_str, topic, pdf_name or 'Konu Bazlı') # Pass explicit name
    # else: (Error already shown inside the try block)
    #     st.error("Seçilen geçmiş sınavın detayları yüklenemedi veya formatı bozuk.")

# Display newly generated exam results if available and no historical exam selected
elif st.session_state.exam_results:
    display_topic = st.session_state.get('last_generated_topic', 'Sinav')
    # Use the source PDF stored during generation (could be None)
    display_pdf_name = st.session_state.get('last_source_pdf', 'Konu Bazlı')
    display_exam(st.session_state.exam_results, display_topic, display_pdf_name or 'Konu Bazlı') # Pass explicit name

# --- Geçmiş Sınavlar (Opsiyonel) ---
st.sidebar.subheader("Geçmiş Sınavlar")
try:
    history = utils.get_exam_history(limit=15) # Increase limit slightly
    if history:
        for entry in history:
            exam_id = entry['id']
            timestamp_str = entry.get('timestamp', 'Bilinmeyen Tarih')
            try:
                timestamp_dt = datetime.fromisoformat(timestamp_str)
                display_time = timestamp_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                display_time = timestamp_str.split('.')[0]

            topic = entry.get('topic', 'Bilinmeyen Konu')
            pdf_name = entry.get('pdf_name', 'Bilinmeyen PDF')
            num_q = entry.get('num_questions', '?')

            # Use columns for better layout within the expander
            with st.sidebar.expander(f"{display_time} - {topic} ({num_q} soru)"):
                col1_info, col2_actions = st.columns([3, 2]) # Adjust ratio as needed

                with col1_info:
                    pdf_display = entry.get('pdf_name', None)
                    if pdf_display:
                        st.caption(f"PDF: {pdf_display}")
                    else:
                        st.caption("Tür: Konu Bazlı")
                    # Removed Session ID and Exam ID display

                with col2_actions:
                    # View Button
                    if st.button("Görüntüle", key=f"view_{exam_id}"):
                        full_exam_data = utils.get_exam_by_id(exam_id)
                        if full_exam_data:
                            # Store the full data to be displayed after rerun
                            st.session_state.exam_to_display = full_exam_data
                            # Clear any newly generated results to avoid conflict
                            st.session_state.exam_results = None
                            st.rerun()
                        else:
                            st.error(f"Sınav ID {exam_id} bulunamadı.")

                    # Delete Button
                    if st.button("Sil", key=f"delete_{exam_id}"):
                        deleted = utils.delete_exam(exam_id)
                        if deleted:
                            st.success(f"Sınav ID {exam_id} silindi.")
                            # Clear displayed exam if it was the one deleted
                            if st.session_state.get('exam_results') and st.session_state.get('last_exam_id') == exam_id:
                                st.session_state.exam_results = None
                                st.session_state.last_exam_id = None
                            if st.session_state.get('exam_to_display') and st.session_state.exam_to_display.get('id') == exam_id:
                                st.session_state.exam_to_display = None
                            st.rerun()
                        else:
                            st.error(f"Sınav ID {exam_id} silinirken hata.")

    else:
        st.sidebar.caption("Geçmiş sınav kaydı bulunamadı.")
except Exception as e:
    st.sidebar.error(f"Geçmiş sınavlar yüklenemedi: {e}") 