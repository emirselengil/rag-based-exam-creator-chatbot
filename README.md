# 🚀 RAG-Based Exam Creator Chatbot

This project is a web application that allows users to automatically generate multiple-choice exams from their own PDF lecture notes or from topics they specify. It is developed using the Retrieval-Augmented Generation (RAG) architecture.

## Project Scope and Problem Solved

Creating exams from relevant materials can be time-consuming for educators, students, or anyone wanting to quickly test their knowledge on a subject. This project aims to automate this process. Using artificial intelligence (Google Gemini), it generates multiple-choice questions and answer keys based on the content of user-uploaded PDF files or a topic specified by the user.

**RAG Architecture:** The project utilizes the RAG (Retrieval-Augmented Generation) approach when creating exams from user-uploaded PDFs:
1.  **Upload and Chunking:** The PDF file is uploaded, and its text content is divided into meaningful chunks.
2.  **Embedding and Storage:** These chunks are converted into vector representations using Google's embedding model and stored in a vector database (ChromaDB). A separate store (identified by a UUID) is created for each PDF.
3.  **Retrieval:** When the user specifies a topic (or selects "Document Overview"), the most relevant text chunks related to that topic are retrieved from the vector database.
4.  **Generation:** The retrieved text chunks (as context) and the user-specified exam parameters (topic, number of questions, difficulty level) are combined with a prompt template and sent to the Google Gemini model.
5.  **Result:** The LLM generates multiple-choice questions and the answer key in JSON format based on the provided context and requirements.

If the user specifies only a topic without uploading a PDF, the RAG steps are skipped, and the LLM generates the exam directly from its general knowledge about that topic.

## Deploy Link

https://rag-based-exam-creator-chatbot.streamlit.app

## Screenshot / Video

![image](https://github.com/EmirSelengil/RAG-based-Exam-Creator-Chatbot/assets/79674066/bd2243b5-30a4-4018-b469-4a0d54103108)

## Features and Use Cases

*   **Exam Generation from PDF:** Create multiple-choice exams from your own lecture notes, articles, or any PDF document.
*   **Topic-Based Exam Generation:** Create exams based on general knowledge about a specific topic (without uploading a PDF).
*   **Customizable Parameters:** Specify the exam topic, number of questions (1-20), and difficulty level (Easy, Medium, Hard).
*   **Source Indication:** For questions generated from PDFs, source information (chunk reference) indicating which text passage the question was derived from is provided.
*   **JSON Format Output:** Generated exams are created in a structured JSON format for easy processing.
*   **Download Options:** Download the generated exams and answer keys in PDF or DOCX (Word) format.
*   **Exam History:** View and access previously created exams.
*   **History Management:** View or delete old exams.

## Technologies Used

*   **Backend & LLM Orchestration:**
    *   Python 3.x
    *   LangChain: Framework for developing LLM applications.
    *   LangChain Community & Google GenAI: Integration with Google Gemini LLM and Embedding models.
*   **Web Framework:**
    *   Streamlit: For creating fast and interactive web applications.
*   **Vector Database:**
    *   ChromaDB: For storing embeddings and performing similarity searches.
    *   SQLite (used by ChromaDB): For database storage.
    *   `pysqlite3-binary`: For compatibility in environments like Streamlit Cloud.
*   **PDF Processing:**
    *   `PyPDFLoader` (LangChain): For loading PDF files.
    *   `RecursiveCharacterTextSplitter` (LangChain): For splitting text into chunks.
*   **Exam Export:**
    *   ReportLab: For creating PDF files.
    *   `python-docx`: For creating DOCX (Word) files.
*   **Other:**
    *   `python-dotenv`: For managing environment variables.

## Local Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/EmirSelengil/RAG-based-Exam-Creator-Chatbot.git
    cd RAG-based-Exam-Creator-Chatbot
    ```
2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # MacOS/Linux
    source .venv/bin/activate
    ```
3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Up API Key:**
    *   Create a file named `.env` in the project root directory.
    *   Inside the `.env` file, add the `GOOGLE_API_KEY` variable with your API key obtained from Google AI Studio or your Google Cloud project:
      ```
      GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      ```
    *   You can get an API key from Google AI Studio ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)). Ensure the "Generative Language API" is enabled.
5.  **Start the Application:**
    ```bash
    streamlit run main.py
    ```
    The application will open in your default web browser.

## Contact

*(You can add your name, email address, or LinkedIn/GitHub profile link here.)*

## Repository Structure

```
.
├── .devcontainer/           # Development environment configuration (optional)
├── .git/                    # Git version control metadata
├── .venv/                   # Python virtual environment (usually excluded by .gitignore)
├── data/
│   └── uploaded_pdfs/       # Location where user-uploaded PDFs are saved
├── db/
│   └── chat_history.sqlite  # SQLite database storing exam history
├── embeddings/
│   └── chroma_db/           # Main directory for ChromaDB vector stores (UUID subdirectories for each PDF)
├── __pycache__/             # Python byte code cache (usually excluded by .gitignore)
├── .env                     # Environment variables like API key (local, excluded by .gitignore)
├── .gitattributes           # Git file attributes
├── .gitignore               # Files/directories ignored by Git
├── main.py                  # Main entry point for the Streamlit application, UI, and flow control
├── rag_pipeline.py          # RAG chain, LLM, embedding, and vector store management
├── requirements.txt         # Project dependencies
├── utils.py                 # Utility functions (DB operations, PDF processing, file saving, export, etc.)
└── README.md                # This file
``` 