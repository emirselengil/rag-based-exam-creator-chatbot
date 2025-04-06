# Utility functions for PDF processing, database interactions, etc. 

import os
import sqlite3
import json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
from langchain.docstore.document import Document # Use this specific import path
import io # For creating in-memory files

# --- Export Libraries ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
from docx import Document as DocxDocument
from docx.shared import Inches

DB_PATH = os.path.join("db", "chat_history.sqlite")
UPLOAD_FOLDER = os.path.join("data", "uploaded_pdfs")

# --- Database Functions ---

def init_db():
    """Initializes the SQLite database and creates/updates the exams table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists (initial schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT, -- Added session ID
            pdf_name TEXT, -- Made nullable
            topic TEXT NOT NULL,
            num_questions INTEGER NOT NULL,
            questions_json TEXT,
            answers_json TEXT
        )
    """)

    # Check and add session_id column if it doesn't exist (for backward compatibility)
    try:
        cursor.execute("PRAGMA table_info(exams)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'session_id' not in columns:
            cursor.execute("ALTER TABLE exams ADD COLUMN session_id TEXT")
            print("Added 'session_id' column to existing exams table.")
    except sqlite3.Error as e:
        print(f"Error checking/altering table schema: {e}")

    conn.commit()
    conn.close()

def save_exam(session_id: str, pdf_name: Optional[str], topic: str, num_questions: int, questions_json: str, answers_json: str):
    """Saves the generated exam details to the database. pdf_name can be None."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO exams (session_id, pdf_name, topic, num_questions, questions_json, answers_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, pdf_name, topic, num_questions, questions_json, answers_json))
    conn.commit()
    conn.close()

def get_exam_history(limit: int = 10) -> List[dict]:
    """Retrieves the most recent exam history from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, session_id, pdf_name, topic, num_questions FROM exams ORDER BY timestamp DESC LIMIT ?", (limit,))
    # Fetch only necessary columns for the list view
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

def get_exam_by_id(exam_id: int) -> dict | None:
    """Retrieves a specific exam record by its ID, including JSON data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exams WHERE id = ?", (exam_id,))
    exam_data = cursor.fetchone()
    conn.close()
    return dict(exam_data) if exam_data else None

def delete_exam(exam_id: int):
    """Deletes an exam record from the database by its ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM exams WHERE id = ?", (exam_id,))
        conn.commit()
        conn.close()
        print(f"Deleted exam record with ID: {exam_id}")
        return True
    except sqlite3.Error as e:
        print(f"Error deleting exam record {exam_id}: {e}")
        return False

# --- PDF Processing Functions ---

def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """Loads a PDF, splits it into chunks, and returns a list of LangChain Documents."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load() # Loads pages as individual documents

        # Combine page contents into one large string first (optional, depends on splitting strategy)
        # full_text = "\n".join([page.page_content for page in pages])
        # Use RecursiveCharacterTextSplitter on the combined text or directly on loaded pages

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Helps in tracing back the source
        )

        # Split the documents loaded by PyPDFLoader
        # PyPDFLoader loads each page as a Document, splitting them further
        chunks = text_splitter.split_documents(pages)

        # Add original PDF filename and page number to metadata for reference
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_pdf"] = os.path.basename(pdf_path)
            original_page_num = chunk.metadata.get('page', 'N/A')
            chunk.metadata["chunk_index"] = i
            # Keep the page number if available for potential source tracking
            if isinstance(original_page_num, int):
                 chunk.metadata["source_page"] = original_page_num + 1
            # Create a more readable source string for the context
            chunk.metadata["source"] = f"{os.path.basename(pdf_path)}{f' - Sayfa {original_page_num + 1}' if isinstance(original_page_num, int) else ''}, Parça {i}"

        return chunks

    except Exception as e:
        print(f"Error loading/splitting PDF {pdf_path}: {e}")
        return [] # Return empty list on error


# --- Utility for saving uploaded file ---
def save_uploaded_file(uploaded_file) -> str:
    """Saves the uploaded Streamlit file to the designated folder.
       Returns the full path to the saved file.
    """
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path 

# --- Exam Export Functions ---

def _parse_exam_data(exam_data_str: str) -> Dict:
    """Parses the JSON string from the LLM or database into a dictionary."""
    try:
        # Try cleaning potential markdown code fences first
        cleaned_response = exam_data_str.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for export: {exam_data_str}")
        # Return a structure indicating error, or raise an exception
        return {"sorular": [], "cevap_anahtari": {}} # Basic fallback

def create_pdf_exam(exam_data_str: str, topic: str, pdf_name: str) -> io.BytesIO:
    """Creates a PDF document for the exam in memory."""
    exam_data = _parse_exam_data(exam_data_str)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['h1']
    title_style.alignment = TA_CENTER
    story.append(Paragraph(f"{topic} Sınavı", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Subtitle (optional - original PDF name)
    sub_title_style = styles['h3']
    sub_title_style.alignment = TA_CENTER
    story.append(Paragraph(f"(Kaynak Belge: {pdf_name})", sub_title_style))
    story.append(Spacer(1, 0.3*inch))

    # Questions
    question_style = styles['Normal']
    question_style.leftIndent = 0.25*inch
    option_style = styles['Normal']
    option_style.leftIndent = 0.5*inch

    for i, q in enumerate(exam_data.get("sorular", [])):
        q_num = q.get('soru_no', i + 1)
        q_text = q.get('soru_metni', 'Soru metni bulunamadı').replace('\n', '<br/>') # Handle newlines
        story.append(Paragraph(f"<b>{q_num}.</b> {q_text}", question_style))
        story.append(Spacer(1, 0.1*inch))

        options = q.get("secenekler", {})
        for key, value in options.items():
            story.append(Paragraph(f"<b>{key})</b> {value.replace('\n', '<br/>')}", option_style))
        story.append(Spacer(1, 0.2*inch))

    # Add page break before answers if there are questions
    if exam_data.get("sorular"):
        story.append(PageBreak())

    # Answer Key
    story.append(Paragraph("Cevap Anahtarı", title_style))
    story.append(Spacer(1, 0.2*inch))

    answers = exam_data.get("cevap_anahtari", {})
    answer_style = styles['Normal']
    answer_style.alignment = TA_LEFT
    answer_lines = [f"<b>{num}:</b> {ans}" for num, ans in answers.items()]
    # Simple list for now, could format into columns if many questions
    for line in answer_lines:
        story.append(Paragraph(line, answer_style))
        story.append(Spacer(1, 0.1*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

def create_docx_exam(exam_data_str: str, topic: str, pdf_name: str) -> io.BytesIO:
    """Creates a DOCX document for the exam in memory."""
    exam_data = _parse_exam_data(exam_data_str)
    document = DocxDocument()

    # Title
    document.add_heading(f"{topic} Sınavı", level=1)
    # Subtitle
    document.add_heading(f"(Kaynak Belge: {pdf_name})", level=3)
    document.add_paragraph()

    # Questions
    for i, q in enumerate(exam_data.get("sorular", [])):
        q_num = q.get('soru_no', i + 1)
        q_text = q.get('soru_metni', 'Soru metni bulunamadı')
        p = document.add_paragraph()
        p.add_run(f"{q_num}. ").bold = True
        p.add_run(q_text)

        options = q.get("secenekler", {})
        for key, value in options.items():
            # Add options with indentation (using tabs or spaces)
            p_opt = document.add_paragraph(style='List Bullet') # Use bullet points
            p_opt.paragraph_format.left_indent = Inches(0.5)
            # Clear the automatic bullet/numbering if needed, add manually
            p_opt.text = f"\t{key}) {value}" # Manual formatting might be simpler
            # Or adjust style for desired look

        document.add_paragraph() # Spacer

    # Add page break before answers if there are questions
    if exam_data.get("sorular"):
        document.add_page_break()

    # Answer Key
    document.add_heading("Cevap Anahtarı", level=1)
    document.add_paragraph()

    answers = exam_data.get("cevap_anahtari", {})
    # Simple list format for answers
    for num, ans in answers.items():
        p_ans = document.add_paragraph()
        p_ans.add_run(f"{num}: ").bold = True
        p_ans.add_run(ans)

    # Save to a BytesIO buffer
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer 